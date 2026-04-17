import sys
import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve
)

# TRIDENT 경로 추가 (사용자 환경에 맞게 유지)
sys.path.append(os.path.join(os.getcwd(), 'TRIDENT'))
from trident.slide_encoder_models import ABMILSlideEncoder

# [필수] 시드 고정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. 모델 클래스 정의
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)
        if return_raw_attention:
            return logits, attn
        return logits

# 2. 데이터셋 클래스 정의 (에러의 원인이었던 부분)
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, fold_col='fold_0', num_features=512):
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        
        self.patient_to_files = {}
        all_files = os.listdir(feats_path)
        for p_id in self.df['patient']:
            matching_files = [
                os.path.join(feats_path, f) for f in all_files 
                if f.startswith(p_id) and f.endswith('.h5')
            ]
            self.patient_to_files[p_id] = matching_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient']
        file_paths = self.patient_to_files.get(patient_id, [])
        
        all_features = []
        for fp in file_paths:
            with h5py.File(fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
        
        features = torch.cat(all_features, dim=0)

        # 테스트 시에는 샘플링 없이 전체 패치를 사용 (전수 검사)
        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available)[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,))
            features = features[indices]

        label = torch.tensor(row["type"], dtype=torch.float32)
        return features, label

# 3. 테스트 실행 및 시각화 함수
def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('./clinical_data_folds.csv')
    feats_path = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
    
    os.makedirs('./test_results', exist_ok=True)
    all_fold_results = []

    # --- 시각화 준비 (루프 밖에서 단 한 번 실행) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for fold in range(5):
        print(f"\n🔍 Testing Fold {fold}...")
        current_fold_col = f'fold_{fold}'
        
        test_dataset = H5Dataset(feats_path, df, split="test", fold_col=current_fold_col)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = BinaryClassificationModel(input_feature_dim=1536).to(device)
        model_path = f'./saved_models/abmil_fold_{fold}_best.pth'
        
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
    save_path = './test_results/combined_fold_curves.png'
    plt.savefig(save_path, dpi=300)
    plt.show() # 결과 확인
    print(f"\n💾 통합 곡선 이미지가 저장되었습니다: {save_path}")

    # 최종 결과 출력
    results_df = pd.DataFrame(all_fold_results)
    if not results_df.empty:
        print(f"\n{'='*20} Final Test Summary {'='*20}")
        print(results_df.to_string(index=False))
        print(f"\nMean AUC: {results_df['AUC'].mean():.4f} ± {results_df['AUC'].std():.4f}")

if __name__ == "__main__":
    test_and_visualize()