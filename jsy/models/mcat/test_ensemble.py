import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix, 
    classification_report, average_precision_score, roc_curve, precision_recall_curve
)
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns 
# 기존에 작성한 모듈 임포트
from modules.mcat_multimodal_dataset import MSI_Multimodal_Dataset, get_label_path
from modules.mcat_model import MCAT_Binary

import logging
import sys

# 1. 로그 설정 (시간, 로그 레벨, 메시지 형식 지정)
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s [%(levelname)s] %(message)s',
    format='%(message)s',
    handlers=[
        logging.FileHandler("abmil_student_test_ensemble.txt"), # 파일 저장
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

def evaluate_test_set(result_path, csv_path, feats_path, npy_path, pkl_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*40}\n🚀 5-Fold Ensemble Inference Start\n{'='*40}")
    # 1. 완전한 Unseen 테스트 데이터셋 로드
    # 생성된 CSV 파일에서 fold_0 컬럼에 'test'라고 마킹된 20%의 데이터를 불러옵니다.
    test_dataset = MSI_Multimodal_Dataset(
        split='test', 
        fold_col='fold_0', # 어느 폴드 컬럼이든 test 셋의 인덱스는 동일합니다.
        csv_path=csv_path,
        feats_path=feats_path,
        npy_path=npy_path,
        pkl_path=pkl_path
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"테스트 환자 수: {len(test_dataset)}명\n")

    model = MCAT_Binary().to(device)
    model.eval() # 평가 모드 전환 (Dropout 등 비활성화)
    # 5폴드의 모델을 전부 테스트하기 위해 반복
    all_labels =[]
    ensemble_probs = None
    for fold in range(5):
    # 2. 모델 초기화 및 학습된 가중치(Best Model) 장착
        model_path = os.path.join(result_path, f"best_model_fold{fold}.pt")
        print(f"로드할 모델 가중치: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        fold_labels = []
        fold_probs = []

        # 3. 실전 인퍼런스 루프        
        with torch.no_grad(): # 테스트 단계이므로 역전파 금지
            pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
            for data_wsi, data_omic, label in pbar:
                data_wsi = data_wsi.to(device)
                data_omic = data_omic.to(device)
                label = label.type(torch.FloatTensor).to(device)
                
                # # 이미지 전용 추론(유전체 더미 데이터 사용)
                # data_omic = torch.zeros_like(data_omic).to(device)

                logits, *_ = model(data_wsi, data_omic)

                # 차원 정리
                logits = logits.squeeze(dim=-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                # 시그모이드 통과하여 0~1 사이 확률값 도출
                probs = torch.sigmoid(logits)
                fold_probs.extend(probs.cpu().numpy())

                if fold == 0:
                    fold_labels.extend(label.cpu().numpy())

        fold_probs = np.array(fold_probs)

        # 확률값 누적
        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_labels = np.array(fold_labels)
        else:
            ensemble_probs += fold_probs
        
    # 평균 = 5개 모델 예측 확률의 총합/5
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
    plt.savefig(os.path.join(result_path, 'MCAT_ensemble_curves.png'), dpi=300)
    plt.show()

    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['MSS (0)', 'MSI (1)'], yticklabels=['MSS (0)', 'MSI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Ensemble Confusion Matrix')
    plt.savefig(os.path.join(result_path, 'MCAT_ensemble_confusion_matrix.png'), dpi=300)
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
    plt.savefig(os.path.join(result_path, 'MCAT_ensemble_histogram.png'), dpi=300)
    plt.show()
    
    # 💡 7. 최종 결과 CSV 한 줄로 요약 저장
    results_dict = {
        'AUROC': [auroc], 'AUPRC': [auprc], 'F1_Score': [f1],
        'Accuracy': [acc], 'Sensitivity': [sensitivity], 'Specificity': [specificity],
        'Threshold': [ensemble_best_thresh]
    }
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(result_path, 'MCAT_ensemble_test_results.csv'), index=False)
    print(f"\n💾 앙상블 최종 결과가 {result_path}에 저장되었습니다.")

if __name__ == '__main__':
    PATIENTS_LABEL = "./data/common_patients.txt"
    CSV_PATH = get_label_path(PATIENTS_LABEL) # get_label_path가 정의되어 있다고 가정
    FEATS_PATH = "/home/team/projects/team_REDCAB/team_project/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath"
    NPY_PATH = "./data/genomic_input_matrix.npy"
    PKL_PATH = "./data/genomic_encoding_states.pkl"
    RESULT_PATH = "./results_msi"
    
    # 파라미터에서 ENSEMBLE_THRESH 제거 (내부에서 자동 계산됨)
    evaluate_test_set(RESULT_PATH, CSV_PATH, FEATS_PATH, NPY_PATH, PKL_PATH)