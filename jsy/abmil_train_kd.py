import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

from utils.abmil_model import BinaryClassificationModel
from utils.h5dataset_full import H5Dataset # kd_path를 처리할 수 있게 수정된 데이터셋 사용
from utils.binary_focal_loss import BinaryFocalLoss
import config as cf

# 설정값 로드
SEED = cf.SEED
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_feats_path()
RESULTS_PATH = cf.get_results_path()
MODEL_PATH = os.path.join(RESULTS_PATH, 'saved_models_kd') # KD 전용 폴더로 분리 권장
KNOWLEDGE_DIR = "./teacher_knowledge" # 앞서 추출한 티처의 해설지 경로

os.makedirs(MODEL_PATH, exist_ok=True)

# 시드 고정
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 💡 지식 증류 복합 손실 함수 (KD Loss)
def kd_loss_fn(s_logits, s_attn, t_logits, t_attn, labels, task_criterion, alpha=0.5, beta=0.5):
    # 1. Task Loss (기존 실제 정답과의 오차)
    task_loss = task_criterion(s_logits, labels)
    
    # 2. Logit Distillation (티처의 확신도 모방)
    logit_loss = F.mse_loss(s_logits, t_logits)
    
    # 3. Attention Distillation (티처의 시선 분포 모방)
    s_attn_log = F.log_softmax(s_attn, dim=-1)
    
    # 텐서 차원 일치화 (1, N)
    if t_attn.dim() == 1:
        t_attn = t_attn.unsqueeze(0)
        
    attn_loss = F.kl_div(s_attn_log, t_attn, reduction='batchmean')
    
    return task_loss + (alpha * logit_loss) + (beta * attn_loss)

# ==========================================
# 5-Fold Cross Validation 실행 루프
# ==========================================

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20

fold_metrics = []

for fold in range(5):
    print(f"\n{'='*20} Starting Fold {fold+1} (Knowledge Distillation) {'='*20}")
    current_fold_col = f'fold_{fold}'
    
    # 💡 티처의 지식 파일 경로 설정
    kd_path = os.path.join(KNOWLEDGE_DIR, f'knowledge_fold{fold}_train.pkl')

    # 1. Fold마다 데이터로더 새롭게 구성
    # Train에는 티처의 지식을 함께 로드하고, Val은 평가만 하므로 로드하지 않음
    train_ds = H5Dataset(split="train", fold_col=current_fold_col, kd_path=kd_path)
    val_ds = H5Dataset(split="val", fold_col=current_fold_col, kd_path=None)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

    # 2. 모델 및 최적화 도구 초기화
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)

    best_val_auprc = 0.0
    best_val_auroc = 0.0
    best_thresh = 0
    patience = 5  
    early_stop_counter = 0

    # 3. Training & Validation Loop
    for epoch in range(num_epochs):
        print(f'Epoch_{epoch} start')

        accumulation_steps = 16
        model.train()
        optimizer.zero_grad()
        
        # 💡 데이터로더에서 티처의 로짓과 어텐션도 함께 받아옴
        for i, (features, coords, labels, t_logits, t_attn) in enumerate(tqdm(train_loader)):
            # 사용자님의 딕셔너리 래핑 방식 유지
            features_dict = {'features': features.to(device)} 
            labels = labels.to(device)
            t_logits = t_logits.to(device)
            t_attn = t_attn.to(device)

            # return_raw_attention=True로 스튜던트의 어텐션 점수 함께 반환
            s_logits, s_attn = model(features_dict, return_raw_attention=True)
            
            # 💡 KD 손실 계산 (alpha와 beta로 증류 강도 조절)
            loss = kd_loss_fn(s_logits, s_attn, t_logits, t_attn, labels, criterion, alpha=1, beta=0)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # 4. Validation Evaluation
        model.eval()
        all_labels, all_probs = [], []
        correct, total = 0, 0

        with torch.no_grad():
            # 검증 시에는 t_logits, t_attn 반환이 없으므로 기존 방식과 동일
            for features, coords, labels in tqdm(val_loader):
                features_dict = {'features': features.to(device)}
                labels = labels.to(device)
                outputs = model(features_dict) # 검증 시에는 raw_attention 불필요
                probs = torch.sigmoid(outputs)
                
                predicted = (outputs > 0).float()  
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_probs.append(probs.cpu().numpy())  
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        try:
            auroc = roc_auc_score(all_labels, all_probs)
            auprc = average_precision_score(all_labels, all_probs)
            
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            best_idx = np.argmax(f1_scores)
            thresh = thresholds[best_idx]
        except ValueError:
            auroc = 0.5
            auprc = 0.0
            thresh = 0.5

        all_preds = (all_probs >= thresh).astype(float)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f'====> Epoch: {epoch:02d} | Valid AUROC: {auroc:.4f} | Valid AUPRC: {auprc:.4f} | Valid F1: {f1:.4f} | Valid Thresh: {thresh:.4f} <====')
        
        # AUPRC를 기준으로 최고 성능 모델 저장
        if auroc > best_val_auroc:
            best_val_auprc = auprc
            best_val_auroc = auroc
            best_thresh = thresh
            checkpoint_path = os.path.join(MODEL_PATH, f"best_model_fold{fold}.pt")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_thresh': best_thresh,
                'auprc': best_val_auprc,
                'auroc': best_val_auroc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"🔥 Fold {fold+1} 최고 성능 갱신! 모델 저장됨 (AUROC: {auroc:.4f}), (AUPRC: {auprc:.4f}), Best Threshold: {thresh:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"{patience} epoch 동안 성능 개선이 없어 Fold {fold+1} 학습 조기 종료")
            break