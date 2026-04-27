import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tqdm import tqdm

from utils.abmil_model import BinaryClassificationModel
from utils.h5dataset_full import H5Dataset
import config as cf

def evaluate_ensemble_test_set():
    # --- 1. 설정 및 경로 초기화 ---
    SEED = cf.SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models_kd')
    TEST_PATH = os.path.join(cf.get_results_path(), 'test_results_kd')
    os.makedirs(TEST_PATH, exist_ok=True)

    print("\n====> ABMIL (KD) 실전 앙상블 테스트 시작 <====")
    
    # --- 2. 테스트 데이터셋 로드 ---
    # Test 셋의 인덱스는 모든 폴드에서 동일하므로 fold_0을 기준으로 하나만 로드합니다.
    test_dataset = H5Dataset(split="test", fold_col='fold_0', kd_path=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"테스트 환자 수: {len(test_dataset)}명\n")

    # 모델 객체 뼈대 생성 (공통 사용)
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.0).to(device)
    model.eval()

    all_labels = []
    ensemble_probs = None
    sum_thresh = 0.0

    # --- 3. 5-Fold 앙상블 인퍼런스 루프 ---
    for fold in range(5): # 0, 1, 2, 3, 4
        model_file = os.path.join(MODEL_PATH, f'best_model_fold{fold}.pt')
        print(f"로드할 모델 가중치: {model_file}")
        
        if not os.path.exists(model_file):
            print(f"⚠️ 경고: {model_file} 파일이 없습니다. 이 폴드는 건너뜁니다.")
            continue

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        best_thresh = checkpoint['best_thresh']
        sum_thresh += best_thresh
        print(f"  - 이 모델의 Best Threshold: {best_thresh:.4f}")

        fold_probs = []
        fold_labels = []

        # 개별 모델 추론
        with torch.no_grad():
            # tqdm은 첫 번째 폴드에서만 출력하여 콘솔 혼잡 방지
            disable_tqdm = False 
            for features, coords, labels in tqdm(test_loader, desc=f"Testing Fold {fold}", disable=disable_tqdm):
                features_dict = {'features': features.to(device)}
                
                outputs = model(features_dict)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.extend(probs)
                
                # 라벨은 한 번만 수집해도 됨 (항상 같은 데이터, 같은 순서)
                if fold == 0:
                    fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs).flatten()

        # 앙상블 확률 누적
        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_labels = np.array(fold_labels).flatten()
        else:
            ensemble_probs += fold_probs

    # --- 4. 최종 앙상블 평균 계산 ---
    ensemble_probs = ensemble_probs / 5.0
    ensemble_thresh = sum_thresh / 5.0

    print(f"\n👉 앙상블에 적용될 평균 Threshold: {ensemble_thresh:.4f}")

    # --- 5. 임계값(Threshold) 적용 및 최종 성능 지표 계산 ---
    auroc = roc_auc_score(all_labels, ensemble_probs)
    auprc = average_precision_score(all_labels, ensemble_probs)
    
    ensemble_preds = (ensemble_probs >= ensemble_thresh).astype(int)
    
    f1 = f1_score(all_labels, ensemble_preds, zero_division=0)
    cm = confusion_matrix(all_labels, ensemble_preds, labels=[0, 1])
    report = classification_report(all_labels, ensemble_preds, target_names=["MSS (0)", "MSI-H (1)"], zero_division=0)

    # --- 6. 최종 결과 출력 ---
    print("\n" + "="*40)
    print("🏆 ABMIL (KD) 5-Fold 앙상블 최종 테스트 결과 🏆")
    print("="*40)
    print(f"Ensemble AUROC : {auroc:.4f}")
    print(f"Ensemble AUPRC : {auprc:.4f}")
    print(f"Ensemble F1    : {f1:.4f}\n")
    
    print("[Confusion Matrix]")
    print(" TN(MSS정답)  FP(MSI오진)")
    print(" FN(MSS오진)  TP(MSI정답)")
    print(cm)
    
    print("\n[Classification Report]")
    print(report)
    
    # 7. 시각화
    plot_ensemble_curves(all_labels, ensemble_probs, auroc, auprc, TEST_PATH)


# --- 기존 evaluate_ensemble_test_set 함수 내부에 지표 계산 후 아래 내용 추가 ---

def plot_ensemble_curves(all_labels, ensemble_probs, auroc, auprc, test_path):
    """
    앙상블 결과에 대한 ROC 및 PR 곡선을 시각화하여 저장합니다.
    """
    # 시각화 스타일 설정
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- 1. Ensemble ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, ensemble_probs)
    
    # 굵고 투명도 없는 진한 파란색 선으로 앙상블 성능 강조
    ax1.plot(fpr, tpr, color='darkblue', lw=3, label=f'Ensemble ROC (AUC = {auroc:.4f})')
    
    # 대각선 (Chance line)
    ax1.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ABMIL (Distilled) Ensemble ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=11, frameon=True)
    ax1.grid(alpha=0.4)

    # --- 2. Ensemble Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(all_labels, ensemble_probs)
    
    # 굵고 진한 주황색 선으로 강조
    ax2.plot(recall, precision, color='darkorange', lw=3, label=f'Ensemble PR (AUPRC = {auprc:.4f})')
    
    # 베이스라인 (MSS 클래스의 비율)
    baseline = np.sum(all_labels) / len(all_labels)
    ax2.axhline(y=baseline, color='grey', lw=1.5, linestyle='--', label=f'Baseline ({baseline:.2f})')
    
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('ABMIL (Distilled) Ensemble Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=11, frameon=True)
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    
    # 결과 저장
    save_path = os.path.join(test_path, 'abmil_ensemble_final_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 앙상블 ROC/PR 곡선 그래프가 저장되었습니다: {save_path}")

# =========================================================
# evaluate_ensemble_test_set 함수 마지막에 아래 한 줄을 추가하세요.
# =========================================================

if __name__ == "__main__":
    evaluate_ensemble_test_set()