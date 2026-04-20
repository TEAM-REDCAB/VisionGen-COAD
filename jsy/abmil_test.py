import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # CM 시각화를 위해 추가
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from abmil_model import BinaryClassificationModel, H5Dataset
import abmil_model as am

# --- 설정 및 경로 ---
SEED = am.SEED
LABEL_PATH = am.get_label_path()
FEATS_PATH = am.get_features_path()
MODEL_PATH = os.path.join(am.get_results_path(), 'saved_models')
TEST_PATH = os.path.join(am.get_results_path(), 'test_results')
os.makedirs(TEST_PATH, exist_ok=True)

# 시드 고정
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(LABEL_PATH)
    
    all_fold_results = []
    
    # --- 평균 ROC 계산을 위한 변수들 ---
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100) # 공통 x축 (0~1 사이 100개 지점)
    
    # --- 합산 혼동 행렬을 위한 변수 ---
    total_cm = np.zeros((2, 2))

    # 시각화 준비 (ROC & PR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for fold in range(5):
        print(f"\n🔍 Testing Fold {fold}...")
        current_fold_col = f'fold_{fold}'
        
        test_dataset = H5Dataset(FEATS_PATH, df, split="test", fold_col=current_fold_col)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 모델 선언 (Train 시와 동일하게 dropout 적용)
        model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
        model_path = os.path.join(MODEL_PATH, f'abmil_fold_{fold}_best.pth')
        
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path} 없음. 스킵.")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_labels, all_probs = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features_dict = {'features': features.to(device)}
                outputs = model(features_dict)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # 지표 계산
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        acc = np.mean(preds == all_labels)
        
        # 1. 혼동 행렬 누적
        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        total_cm += cm
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        all_fold_results.append({
            'Fold': fold, 'AUC': auc, 'AUPRC': auprc, 
            'Accuracy': acc, 'Sensitivity': sensitivity, 'Specificity': specificity
        })

        # 2. ROC 곡선 보간(Interpolation) 및 저장
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr) # 공통 x축에 맞춰 y값 추출
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)

        # 개별 폴드 곡선 그리기 (투명도 조절)
        ax1.plot(fpr, tpr, color=colors[fold], lw=1.5, alpha=0.3, label=f'Fold {fold} (AUC = {auc:.4f})')
        
        # PR Curve 그리기
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        ax2.plot(recall, precision, color=colors[fold], lw=1.5, alpha=0.3)
        
        print(f"✅ Fold {fold} 완료")

    # --- 3. 평균 ROC 및 표준편차 영역(Shaded Area) 그리기 ---
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # 평균 선 (굵게)
    ax1.plot(mean_fpr, mean_tpr, color='black', lw=3, label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    
    # 표준편차 영역 (그림자)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

    # --- 그래프 마무리 설정 ---
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
    plt.savefig(os.path.join(TEST_PATH, 'combined_fold_curves.png'), dpi=300)
    plt.show()

    # --- 4. 합산 혼동 행렬(Confusion Matrix) 시각화 ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['MSS (0)', 'MSI (1)'], yticklabels=['MSS (0)', 'MSI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Aggregated Confusion Matrix (All 5 Folds)')
    plt.savefig(os.path.join(TEST_PATH, 'total_confusion_matrix.png'), dpi=300)
    plt.show()

    # 최종 결과 요약 및 저장
    results_df = pd.DataFrame(all_fold_results)
    if not results_df.empty:
        print(f"\n{'='*20} Final Test Summary {'='*20}")
        print(results_df.to_string(index=False))
        print(f"\nMean AUC: {results_df['AUC'].mean():.4f} ± {results_df['AUC'].std():.4f}")
    
    results_df.to_csv(os.path.join(TEST_PATH, 'test_results.csv'), index=False)
    print(f"\n💾 모든 결과가 {TEST_PATH}에 저장되었습니다.")

if __name__ == "__main__":
    test_and_visualize()