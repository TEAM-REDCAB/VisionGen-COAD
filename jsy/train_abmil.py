import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.join(os.getcwd(), 'TRIDENT'))

from trident.slide_encoder_models import ABMILSlideEncoder

# Set deterministic behavior
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


class H5Dataset(Dataset):
    # [수정됨] fold_col 파라미터 추가하여 동적으로 Fold 컬럼을 바라보도록 변경
    def __init__(self, feats_path, df, split, fold_col='fold_0', num_features=512):
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        
        # 초기화 단계에서 환자별 파일 매핑 딕셔너리 생성 (속도 최적화)
        self.patient_to_files = {}
        all_files = os.listdir(feats_path)
        
        for p_id in self.df['patient']:
            # 환자 ID(12자리)로 시작하고 .h5로 끝나는 모든 파일 찾기
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
        
        if not file_paths:
            raise FileNotFoundError(f"환자 {patient_id}에 대한 .h5 파일을 찾을 수 없습니다.")

        # 여러 슬라이드의 피처를 리스트에 담은 후 하나로 병합
        all_features = []
        for fp in file_paths:
            with h5py.File(fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
        
        features = torch.cat(all_features, dim=0)

        # 학습 시 고정된 개수로 샘플링
        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))  # Oversampling
            features = features[indices]

        label = torch.tensor(row["type"], dtype=torch.float32)
        return features, label


# ==========================================
# 5-Fold Cross Validation 실행 루프
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('./clinical_data_folds.csv')
feats_path = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
batch_size = 8
num_epochs = 20

# 모델 가중치 저장 디렉토리 생성
os.makedirs('./saved_models', exist_ok=True)

# 5-Fold 성능 결과를 모아둘 리스트
fold_metrics = []

for fold in range(5):
    print(f"\n{'='*20} Starting Fold {fold} {'='*20}")
    current_fold_col = f'fold_{fold}'

    # 1. Fold마다 데이터로더 새롭게 구성 (train, val)
    train_loader = DataLoader(
        H5Dataset(feats_path, df, split="train", fold_col=current_fold_col), 
        batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED)
    )
    val_loader = DataLoader(
        H5Dataset(feats_path, df, split="val", fold_col=current_fold_col), 
        batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED)
    )

    # 2. Fold마다 모델과 옵티마이저 완전히 새로 초기화 (데이터 누수 방지)
    model = BinaryClassificationModel(input_feature_dim=1536).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-4)

    # 최고 성능을 추적하기 위한 변수 초기화
    best_val_auc = 0.0
    best_metrics = {}

    # 3. Training & Validation Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        for features, labels in train_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)

        # 4. Validation Evaluation (해당 Fold의 검증 성능 측정)
        model.eval()
        all_labels, all_probs = [], []
        correct, total = 0, 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = {'features': features.to(device)}, labels.to(device)
                outputs = model(features) 
                probs = torch.sigmoid(outputs)
                
                predicted = (outputs > 0).float()  
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_probs.append(probs.cpu().numpy())  
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 현재 Epoch 지표 계산
        val_auc = roc_auc_score(all_labels, all_probs)
        val_auprc = average_precision_score(all_labels, all_probs)
        val_accuracy = correct / total

        print(f"[Fold {fold} | Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        # --- 모델 저장 로직 (Best Model Checkpointing) ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {'Fold': fold, 'AUC': val_auc, 'AUPRC': val_auprc, 'Accuracy': val_accuracy}
            
            # 모델 가중치 저장 (안전하게 state_dict만 저장)
            save_path = f'./saved_models/abmil_fold_{fold}_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  --> [New Best!] Model saved to {save_path} (AUC: {best_val_auc:.4f})")

    # 모든 Epoch 종료 후, 해당 Fold의 최고 성능 지표를 최종 결과에 추가
    fold_metrics.append(best_metrics)
    print(f"Fold {fold} Finished. Best AUC: {best_val_auc:.4f}")
    
# ==========================================
# 5-Fold 최종 요약 리포트
# ==========================================
print(f"\n{'='*20} 5-Fold Cross Validation Summary (Best Epochs) {'='*20}")
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df.to_string(index=False))
print("-" * 50)
print(f"Mean Best AUC:     {metrics_df['AUC'].mean():.4f} ± {metrics_df['AUC'].std():.4f}")
print(f"Mean Best AUPRC:   {metrics_df['AUPRC'].mean():.4f} ± {metrics_df['AUPRC'].std():.4f}")
print(f"Mean Best Acc:     {metrics_df['Accuracy'].mean():.4f} ± {metrics_df['Accuracy'].std():.4f}")