import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import gigapath_model as gm 
from gigapath.classification_head import ClassificationHead

SEED = gm.SEED
RESULTS_PATH = gm.get_results_path()
MODEL_PATH = os.path.join(RESULTS_PATH, 'saved_models')

# Set deterministic behavior
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 5-Fold Cross Validation 실행 루프 (ViT + Flash Attn 버전)
# ==========================================

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [주의] RTX 3060 12GB VRAM 제약으로 인해 ViT에서는 batch_size를 과감히 줄여야 합니다.
batch_size = 1 
num_epochs = 10
# OOM 발생 시 num_features(샘플링 패치 수)를 4096 -> 2048 -> 1024 순으로 줄여보세요.
train_num_features = 4096 

fold_metrics = []

for fold in range(5):
    print(f"\n{'='*20} Starting Fold {fold} {'='*20}")
    current_fold_col = f'fold_{fold}'

    # 1. Fold마다 데이터로더 구성 (Coords 경로 추가)
    train_ds = gm.H5Dataset(split="train", fold_col=current_fold_col, num_features=train_num_features)
    # Val은 전체 패치를 사용하므로 VRAM 관리를 위해 num_features 제한을 두거나, 
    # OOM이 터진다면 Val 데이터셋에도 적절한 num_features 제한이 필요할 수 있습니다.
    val_ds = gm.H5Dataset(split="val", fold_col=current_fold_col)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

    # 2. ClassificationHead (ViT) 모델 초기화
    model = ClassificationHead(
        input_dim=1536,
        latent_dim=768,
        feat_layer="5-11",
        n_classes=1, # BinaryFocalLoss와 호환되도록 1로 설정
        freeze=False
    ).to(device)
    
    criterion = gm.BinaryFocalLoss(alpha=0.75, gamma=2).to(device)
    # ViT 모델에 더 적합한 AdamW 사용 및 Learning Rate 미세 조정 (ABMIL보다 조금 낮게 시작)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Mixed Precision 학습을 위한 Scaler 초기화 (메모리 절약 및 속도 향상)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = 0.0
    best_metrics = {}

    # 3. Training & Validation Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        
        # DataLoader에서 features, coords, labels 3가지를 반환하도록 변경됨
        for features, coords, labels in train_loader:
            features = features.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # AMP 적용 블록
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 출력을 [batch, 1]에서 [batch] 형태로 맞추기 위해 squeeze 적용
                outputs = model(features, coords).squeeze(1)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)

        # 4. Validation Evaluation
        torch.cuda.empty_cache()
        model.eval()
        all_labels, all_probs = [], []
        correct, total = 0, 0

        with torch.no_grad():
            for features, coords, labels in val_loader:
                features = features.to(device)
                coords = coords.to(device)
                labels = labels.to(device)
                
                # Validation 시에도 메모리 절약을 위해 AMP 적용
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(features, coords).squeeze(1)
                    probs = torch.sigmoid(outputs)
                
                predicted = (outputs > 0).float()  
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_probs.append(probs.cpu().numpy())  
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        val_auc = roc_auc_score(all_labels, all_probs)
        val_auprc = average_precision_score(all_labels, all_probs)
        val_accuracy = correct / total

        print(f"[Fold {fold} | Epoch {epoch+1:02d}/{num_epochs}] Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        # --- 모델 저장 로직 ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = {'Fold': fold, 'AUC': val_auc, 'AUPRC': val_auprc, 'Accuracy': val_accuracy}
            
            os.makedirs(MODEL_PATH, exist_ok=True)
            save_path = os.path.join(MODEL_PATH, f'vit_gigapath_fold_{fold}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  --> [New Best!] Model saved to {save_path} (AUC: {best_val_auc:.4f})")

    fold_metrics.append(best_metrics)
    print(f"Fold {fold} Finished. Best AUC: {best_val_auc:.4f}")

    # [추가] 현재 폴드 모델과 옵티마이저 명시적 삭제
    del model
    if 'optimizer' in locals(): del optimizer
    if 'scaler' in locals(): del scaler
    
    # [추가] 가비지 컬렉션 및 GPU 캐시 강제 비우기
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"🧹 Fold {fold} 메모리 정리 완료.")
    
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