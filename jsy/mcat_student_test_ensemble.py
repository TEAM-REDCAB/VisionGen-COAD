import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
from utils.mcat_student_model import MCAT_Student

# 💡 훈련 시 사용했던 전체 피처 데이터셋(h5dataset_full)으로 import 맞춤
from utils.h5dataset_full import H5Dataset 
import config as cf

import logging
import sys

# 1. 로그 설정 (시간, 로그 레벨, 메시지 형식 지정)
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s [%(levelname)s] %(message)s',
    format='%(message)s',
    handlers=[
        logging.FileHandler("mcat_student_test_ensemble.txt"), # 파일 저장
        logging.StreamHandler() # 콘솔에도 동시에 출력
    ]
)

# 2. print 문을 logging으로 리다이렉트하는 클래스
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip(): # 빈 줄이 아닐 때만 기록
            self.level(message.strip())

    def flush(self):
        pass

# 3. 시스템의 표준 출력(stdout)과 에러(stderr)를 logging에 연결
sys.stdout = LoggerWriter(logging.info)
# sys.stderr = LoggerWriter(logging.error)

# --- 설정 및 경로 ---
SEED = cf.SEED

# 💡 1. 모델 로드 경로를 KD 훈련 가중치가 있는 곳으로 변경
MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models_mcat_kd')

# 💡 2. 결과 덮어쓰기 방지를 위해 KD 전용 테스트 결과 폴더 생성
TEST_PATH = os.path.join(cf.get_results_path(), 'test_results_mcat_kd_ensemble')
os.makedirs(TEST_PATH, exist_ok=True)

# 시드 고정
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*40}\n🚀 5-Fold Ensemble Inference Start\n{'='*40}")
    
    # 💡 1. 데이터 로드 (매우 중요)
    # 앙상블은 환자의 순서가 무조건 100% 일치해야 확률을 더할 수 있습니다.
    # 따라서 5번 루프를 돌 때마다 데이터를 부르지 않고, 바깥에서 딱 1번만 로드합니다.
    test_dataset = H5Dataset(split="test", fold_col='fold_0', kd_path=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_labels = []
    ensemble_probs = None
    
    # 💡 2. 5개의 폴드 모델을 순회하며 확률값 누적하기
    for fold in range(5):
        print(f"\n🔍 Running Inference: Model Fold {fold}...")
        model = MCAT_Student().to(device)
        model_path = os.path.join(MODEL_PATH, f'best_model_fold{fold}.pt')
        
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path} 없음. 스킵.")
            continue
            
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 이전 모델들의 best_thresh는 더 이상 불러오지 않습니다. (완전 무시)
        model.eval()
        
        fold_probs = []
        fold_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"[Fold {fold}]", leave=False, dynamic_ncols=True)
            for features, coords, labels in pbar:
                features = features.to(device)
                
                # 예측 진행
                logits, _, _ = model(features)
                logits = logits.squeeze(dim=-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                    
                # 0~1 사이의 확률값 도출
                probs = torch.sigmoid(logits)
                fold_probs.extend(probs.cpu().numpy())
                
                # 라벨은 환자 순서가 같으므로 첫 번째 루프(fold 0)에서만 한 번 수집합니다.
                if fold == 0: 
                    fold_labels.extend(labels.cpu().numpy())
                    
        # 리스트를 Numpy 배열로 변환
        fold_probs = np.array(fold_probs)
        
        # 앙상블 확률 누적 합산
        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_labels = np.array(fold_labels)
        else:
            ensemble_probs += fold_probs
            
    # 💡 3. 최종 확률 평균 계산 (단순 합산을 5로 나눔)
    ensemble_probs = ensemble_probs / 5.0
    
    # 💡 4. 앙상블 확률만을 위한 새로운 Youden's Index Threshold 계산
    fpr, tpr, roc_thresholds = roc_curve(all_labels, ensemble_probs)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    ensemble_best_thresh = roc_thresholds[best_idx]
    
    print(f"\n✨ 새롭게 계산된 앙상블 최적 Threshold (Youden's Index): {ensemble_best_thresh:.4f}")
    
    # 💡 5. 앙상블 최종 지표 계산
    auroc = roc_auc_score(all_labels, ensemble_probs)
    auprc = average_precision_score(all_labels, ensemble_probs)
    preds = (ensemble_probs >= ensemble_best_thresh).astype(int)
    acc = np.mean(preds == all_labels)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # --- 결과 출력 ---
    print("\n" + "="*40)
    print("========== 🏆 앙상블 최종 테스트 결과 ==========")
    print("="*40)
    print(f"Ensemble AUROC     : {auroc:.4f}")
    print(f"Ensemble AUPRC     : {auprc:.4f}")
    print(f"Ensemble F1-Score  : {f1:.4f}")
    print(f"Ensemble Threshold : {ensemble_best_thresh:.4f} ")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Sensitivity(Recall): {sensitivity:.4f}")
    print(f"Specificity        : {specificity:.4f}")
    
    print("\n[Confusion Matrix]")
    print(" TN(MSS맞춤)  FP(MSI로오해)")
    print(" FN(MSS로오해) TP(MSI맞춤)")
    print(cm)
    
    report = classification_report(all_labels, preds, target_names=["MSS (0)", "MSI-H (1)"], zero_division=0)
    print("\n[Classification Report]")
    print(report)

    # 💡 6. 시각화 (앙상블된 '단 하나의 완벽한 선'만 그립니다)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC Curve 그리기
    ax1.plot(fpr, tpr, color='red', lw=2, label=f'Ensemble ROC (AUC = {auroc:.4f})')
    ax1.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    # 최적의 Threshold 위치에 별표(★) 표시
    ax1.scatter(fpr[best_idx], tpr[best_idx], marker='*', color='blue', s=200, label=f'Best Thresh ({ensemble_best_thresh:.4f})', zorder=5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Ensemble ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve 그리기
    precision, recall, _ = precision_recall_curve(all_labels, ensemble_probs)
    ax2.plot(recall, precision, color='blue', lw=2, label=f'Ensemble PR (AUC = {auprc:.4f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Ensemble Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_PATH, 'MCAT_kd_ensemble_curves.png'), dpi=300)
    plt.show()

    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['MSS (0)', 'MSI (1)'], yticklabels=['MSS (0)', 'MSI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Ensemble Confusion Matrix')
    plt.savefig(os.path.join(TEST_PATH, 'MCAT_kd_ensemble_confusion_matrix.png'), dpi=300)
    plt.show()

    # 히스토그램 시각화 (앙상블 확률 기준)
    plt.figure(figsize=(10, 6))
    mss_probs = ensemble_probs[all_labels == 0]
    msi_probs = ensemble_probs[all_labels == 1]
    
    sns.histplot(mss_probs, bins=50, color='blue', alpha=0.6, label='MSS (0)', kde=True)
    sns.histplot(msi_probs, bins=50, color='red', alpha=0.6, label='MSI-H (1)', kde=True)
    
    # 히스토그램 위에 새로운 Threshold를 검은 점선으로 표시하여 직관성 확보
    plt.axvline(ensemble_best_thresh, color='black', linestyle='--', lw=2, label=f'Threshold ({ensemble_best_thresh:.4f})')
    
    plt.xlabel('Ensemble Predicted Probability')
    plt.ylabel('Count of Patients')
    plt.title('Distribution of Ensemble Probabilities')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_PATH, 'MCAT_kd_ensemble_histogram.png'), dpi=300)
    plt.show()
    
    # 💡 7. 최종 결과 CSV 한 줄로 요약 저장
    results_dict = {
        'AUROC': [auroc], 'AUPRC': [auprc], 'F1_Score': [f1],
        'Accuracy': [acc], 'Sensitivity': [sensitivity], 'Specificity': [specificity],
        'Threshold': [ensemble_best_thresh]
    }
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(TEST_PATH, 'MCAT_kd_ensemble_test_results.csv'), index=False)
    print(f"\n💾 앙상블 최종 결과가 {TEST_PATH}에 저장되었습니다.")

if __name__ == "__main__":
    test_and_visualize()