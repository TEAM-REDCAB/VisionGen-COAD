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
from utils.abmil_model import BinaryClassificationModel

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
        logging.FileHandler("abmil_kd.txt"), # 파일 저장
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
MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models_kd')

# 💡 2. 결과 덮어쓰기 방지를 위해 KD 전용 테스트 결과 폴더 생성
TEST_PATH = os.path.join(cf.get_results_path(), 'test_results_kd')
os.makedirs(TEST_PATH, exist_ok=True)

# 시드 고정
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_fold_results = []
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100) 
    
    total_cm = np.zeros((2, 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for fold in range(5):
        print(f"\n🔍 Testing Fold {fold} (KD Model)...")
        current_fold_col = f'fold_{fold}'
        
        # Test 셋 로드 (KD 파일 경로 불필요)
        test_dataset = H5Dataset(split="test", fold_col=current_fold_col, kd_path=None)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
        model_path = os.path.join(MODEL_PATH, f'best_model_fold{fold}.pt')
        
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path} 없음. 스킵.")
            continue
            
        # 오타(chechpoint) 수정
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_thresh = checkpoint['best_thresh']
        model.eval()

        all_labels, all_probs = [], []
        with torch.no_grad():
            for features, coords, labels in tqdm(test_loader):
                # 딕셔너리 래핑 방식 유지 (완벽히 작동하는 부분)
                features_dict = {'features': features.to(device)}
                outputs = model(features_dict)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        preds = (all_probs >= best_thresh).astype(int)
        acc = np.mean(preds == all_labels)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        total_cm += cm
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        all_fold_results.append({
            'Fold': fold, 'AUROC': auroc, 'AUPRC': auprc, 
            'Accuracy': acc, 'Sensitivity': sensitivity, 'Specificity': specificity, 
            'F1_Score': f1, 'Threshold': best_thresh
        })

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr) 
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auroc)

        ax1.plot(fpr, tpr, color=colors[fold], lw=1.5, alpha=0.3, label=f'Fold {fold} (AUC = {auroc:.4f})')
        
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        ax2.plot(recall, precision, color=colors[fold], lw=1.5, alpha=0.3)
        
        print("\n" + "="*40)
        print("========== 🏆 최종 테스트 결과 ==========")
        print("="*40)
        print(f"AUROC     : {auroc:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"Threshold : {best_thresh:.4f} ")
        print("\n[Confusion Matrix]")
        print(" TN(MSS맞춤)  FP(MSI로오해)")
        print(" FN(MSS로오해) TP(MSI맞춤)")
        print(cm)
        report = classification_report(all_labels, preds, target_names=["MSS (0)", "MSI-H (1)"], zero_division=0)
        print("\n[Classification Report]")
        print(report)
        
        print(f"✅ Fold {fold} 완료")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    ax1.plot(mean_fpr, mean_tpr, color='black', lw=3, label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

    ax1.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Mean ROC Curves with Std. Dev.')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    # 💡 그래프 저장 이름 변경
    plt.savefig(os.path.join(TEST_PATH, 'abmil_kd_combined_fold_curves_kd.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(total_cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['MSS (0)', 'MSI (1)'], yticklabels=['MSS (0)', 'MSI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('ABMIL (KD) Aggregated Confusion Matrix (All 5 Folds)')
    
    # 💡 CM 저장 이름 변경
    plt.savefig(os.path.join(TEST_PATH, 'abmil_kd_total_confusion_matrix_kd.png'), dpi=300)
    plt.show()

    results_df = pd.DataFrame(all_fold_results)
    if not results_df.empty:
        print(f"\n{'='*20} Final Test Summary {'='*20}")
        print(results_df.to_string(index=False))
        print(f"\nMean AUC: {results_df['AUROC'].mean():.4f} ± {results_df['AUROC'].std():.4f}")
    
    # 💡 CSV 저장 이름 변경
    results_df.to_csv(os.path.join(TEST_PATH, 'abmil_kd_test_results_kd.csv'), index=False)
    print(f"\n💾 모든 결과가 {TEST_PATH}에 저장되었습니다.")

if __name__ == "__main__":
    test_and_visualize()