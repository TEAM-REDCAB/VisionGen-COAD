import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from transmil_model import BinaryClassificationModel, H5Dataset
import transmil_model as tm

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(tm.get_label_path())
    MODEL_PATH = os.path.join(tm.get_results_path(), 'saved_models_transmil')
    
    # TransMIL 구조적 고정 크기
    MAX_PATCHES = tm.MAX_PATCHES 

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    total_cm = np.zeros((2, 2))
    
    plt.figure(figsize=(10, 8))

    for fold in range(5):
        print(f"🔍 Testing Fold {fold}...")
        
        # 모델 초기화 및 로드
        model = BinaryClassificationModel(input_feature_dim=1536).to(device)
        best_model_path = os.path.join(MODEL_PATH, f'transmil_fold_{fold}_best.pth')
        
        if not os.path.exists(best_model_path):
            print(f"  ⚠️ {best_model_path}을 찾을 수 없어 스킵합니다.")
            continue
            
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # 데이터셋 (split="test"는 이제 전수 조사를 위해 샘플링 없이 반환함)
        test_ds = H5Dataset(tm.get_features_path(), df, split="test", fold_col=f'fold_{fold}')
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        probs, labels = [], []
        with torch.no_grad():
            for f, l in test_loader:
                # f shape: [1, total_patches, 1536]
                full_features = f.squeeze(0) # [total_patches, 1536]
                total_p = full_features.shape[0]
                
                chunk_logits = []
                # --- [핵심: Chunking Inference] ---
                for i in range(0, total_p, MAX_PATCHES):
                    chunk = full_features[i : i + MAX_PATCHES]
                    
                    # 마지막 조각 패딩 처리 (PPEG 동작 보장)
                    if chunk.shape[0] < MAX_PATCHES:
                        pad_size = MAX_PATCHES - chunk.shape[0]
                        chunk = torch.cat([chunk, torch.zeros(pad_size, 1536)], dim=0)
                    
                    # 추론
                    out = model({'features': chunk.unsqueeze(0).to(device)})
                    chunk_logits.append(out.item())
                
                # 해당 환자의 모든 구역 결과를 평균내어 최종 확률 계산
                avg_logit = np.mean(chunk_logits)
                final_prob = torch.sigmoid(torch.tensor(avg_logit)).item()
                
                probs.append(final_prob)
                labels.append(l.item())

        # 결과 계산
        auc = roc_auc_score(labels, probs)
        aucs.append(auc)
        print(f"  ✅ Fold {fold} AUC: {auc:.4f}")
        
        # ROC 보간 및 시각화 데이터 축적
        fpr, tpr, _ = roc_curve(labels, probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold} (AUC={auc:.3f})')
        
        # Confusion Matrix 누적 (임계값 0.5 기준)
        preds = [1 if p >= 0.5 else 0 for p in probs]
        total_cm += confusion_matrix(labels, preds, labels=[0, 1])

    # --- 시각화 마무리 ---
    # 1. 통합 ROC Curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    plt.plot(mean_fpr, mean_tpr, color='black', lw=2, label=f'Mean ROC (AUC={mean_auc:.3f})')
    
    std_tpr = np.std(tprs, axis=0)
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='grey', alpha=0.2)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.5)
    plt.title('TransMIL Cross-Validation ROC (Full-Slide)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    # 2. 통합 Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['MSS', 'MSI'], yticklabels=['MSS', 'MSI'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('TransMIL Aggregated Confusion Matrix (Full-Slide)')
    plt.show()

if __name__ == "__main__":
    test()