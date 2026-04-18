import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve
)
from config import BinaryClassificationModel, H5Dataset
import config as cf

SEED = cf.SEED
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_features_path()
MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models')
TEST_PATH = os.path.join(cf.get_results_path(), 'test_results')
os.makedirs(TEST_PATH, exist_ok=True)


# [필수] 시드 고정
np.random.seed(SEED)
torch.manual_seed(SEED)

# 3. 테스트 실행 및 시각화 함수
def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(LABEL_PATH)
    
    all_fold_results = []

    # --- 시각화 준비 (루프 밖에서 단 한 번 실행) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for fold in range(5):
        print(f"\n🔍 Testing Fold {fold}...")
        current_fold_col = f'fold_{fold}'
        
        test_dataset = H5Dataset(FEATS_PATH, df, split="test", fold_col=current_fold_col)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = BinaryClassificationModel(input_feature_dim=1536).to(device)
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

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 지표 계산
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        acc = np.mean(preds == all_labels)
        all_fold_results.append({'Fold': fold, 'AUC': auc, 'AUPRC': auprc, 'Accuracy': acc})

        # 1. ROC Curve 그리기
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        ax1.plot(fpr, tpr, color=colors[fold], lw=2, label=f'Fold {fold} (AUC = {auc:.4f})')

        # 2. PR Curve 그리기
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        ax2.plot(recall, precision, color=colors[fold], lw=2, label=f'Fold {fold} (AUPRC = {auprc:.4f})')
        
        print(f"✅ Fold {fold} -> AUC: {auc:.4f} | AUPRC: {auprc:.4f}")

    # --- 그래프 마무리 설정 ---
    ax1.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Combined ROC Curves')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Combined Precision-Recall Curves')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(TEST_PATH, 'combined_fold_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.show() # 결과 확인
    print(f"\n💾 통합 곡선 이미지가 저장되었습니다: {save_path}")

    # 최종 결과 출력
    results_df = pd.DataFrame(all_fold_results)
    if not results_df.empty:
        print(f"\n{'='*20} Final Test Summary {'='*20}")
        print(results_df.to_string(index=False))
        print(f"\nMean AUC: {results_df['AUC'].mean():.4f} ± {results_df['AUC'].std():.4f}")
    results_df.to_csv(os.path.join(TEST_PATH, 'test_results.csv'), index=False)

if __name__ == "__main__":
    test_and_visualize()