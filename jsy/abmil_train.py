import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from abmil_model import BinaryClassificationModel, H5Dataset
import abmil_model as am

SEED = am.SEED
LABEL_PATH = am.get_label_path()
FEATS_PATH = am.get_features_path()
RESULTS_PATH = am.get_results_path()
MODEL_PATH = os.path.join(RESULTS_PATH, 'saved_models')


# Set deterministic behavior
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 5-Fold Cross Validation 실행 루프
# ==========================================

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(LABEL_PATH)
batch_size = 8
num_epochs = 20

# 5-Fold 성능 결과를 모아둘 리스트
fold_metrics = []

for fold in range(5):
    print(f"\n{'='*20} Starting Fold {fold} {'='*20}")
    current_fold_col = f'fold_{fold}'

    # 1. Fold마다 데이터로더 새롭게 구성 (train:4096샘플링, val:전체)
    train_ds = H5Dataset(FEATS_PATH, df, split="train", fold_col=current_fold_col, num_features=4096)
    val_ds = H5Dataset(FEATS_PATH, df, split="val", fold_col=current_fold_col)
    # Val은 패치 개수가 환자마다 달라 batch_size=1 필수
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

    # 2. Fold마다 모델과 옵티마이저 완전히 새로 초기화 (데이터 누수 방지)
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
    # 불균형 데이터를 위해 Focal Loss 적용
    criterion = am.BinaryFocalLoss(alpha=0.75, gamma=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)

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
            os.makedirs(MODEL_PATH, exist_ok=True)
            save_path = os.path.join(MODEL_PATH, f'abmil_fold_{fold}_best.pth')
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
metrics_df.to_csv(os.path.join(MODEL_PATH, 'metrics.txt'), index=False)