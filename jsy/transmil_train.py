import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transmil_model import BinaryClassificationModel, H5Dataset
import transmil_model as tm

# 설정
SEED = tm.SEED
LABEL_PATH = tm.get_label_path()
FEATS_PATH = tm.get_features_path()
MODEL_SAVE_PATH = os.path.join(tm.get_results_path(), 'saved_models_transmil')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(LABEL_PATH)
    
    # TransMIL의 구조적 고정 크기 (64x64)
    MAX_PATCHES = tm.MAX_PATCHES 

    for fold in range(5):
        print(f"\n{'='*20} Starting TransMIL Fold {fold} {'='*20}")
        
        # 1. 데이터로더 설정
        # Train은 4096개로 샘플링됨, Val은 전체 패치가 나옴
        train_ds = H5Dataset(FEATS_PATH, df, split="train", fold_col=f'fold_{fold}')
        val_ds = H5Dataset(FEATS_PATH, df, split="val", fold_col=f'fold_{fold}')
        
        # Val은 패치 개수가 환자마다 달라 batch_size=1 필수
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True) 
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # 2. 모델 및 최적화 설정
        model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
        
        # 사용자가 정의한 Focal Loss 적용
        criterion = tm.BinaryFocalLoss(alpha=0.75, gamma=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        best_auc = 0
        for epoch in range(20):
            # --- [Train Loop] ---
            model.train()
            total_loss = 0
            for features, labels in train_loader:
                optimizer.zero_grad()
                # Train 시 features shape: [1, 4096, 1536]
                outputs = model({'features': features.to(device)})
                loss = criterion(outputs, labels.float().to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # --- [Validation Loop: 전수 조사 & Chunking] ---
            model.eval()
            val_probs, val_labels = [], []
            
            with torch.no_grad():
                for f, l in val_loader:
                    # f shape: [1, total_patches, 1536]
                    full_features = f.squeeze(0) # [total_patches, 1536]
                    total_p = full_features.shape[0]
                    
                    chunk_logits = []
                    # 메모리 절약을 위해 슬라이드를 조각내서 추론
                    for i in range(0, total_p, MAX_PATCHES):
                        chunk = full_features[i : i + MAX_PATCHES]
                        
                        # 마지막 조각이 4096보다 작을 경우 패딩 (PPEG 동작 보장)
                        if chunk.shape[0] < MAX_PATCHES:
                            pad_size = MAX_PATCHES - chunk.shape[0]
                            chunk = torch.cat([chunk, torch.zeros(pad_size, 1536)], dim=0)
                        
                        # 추론 [1, 4096, 1536] -> [1]
                        out = model({'features': chunk.unsqueeze(0).to(device)})
                        chunk_logits.append(out.item())
                    
                    # 해당 환자의 모든 구역 logit을 평균내어 최종 확률 계산
                    avg_logit = np.mean(chunk_logits)
                    prob = torch.sigmoid(torch.tensor(avg_logit)).item()
                    
                    val_probs.append(prob)
                    val_labels.append(l.item())
            
            # 지표 계산
            val_auc = roc_auc_score(val_labels, val_probs)
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
            
            # 최고 성능 모델 저장
            if val_auc > best_auc:
                best_auc = val_auc
                save_path = os.path.join(MODEL_SAVE_PATH, f'transmil_fold_{fold}_best.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  ⭐ New Best Model Saved! (AUC: {best_auc:.4f})")
            
            scheduler.step()

if __name__ == "__main__":
    train()