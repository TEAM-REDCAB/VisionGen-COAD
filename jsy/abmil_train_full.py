import os
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

from utils.abmil_model import BinaryClassificationModel
from utils.h5dataset_full import H5Dataset
from utils.binary_focal_loss import BinaryFocalLoss
import config as cf

SEED = cf.SEED
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_feats_path()
RESULTS_PATH = cf.get_results_path()
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
    print(f"\n{'='*20} Starting Fold {fold+1} {'='*20}")
    current_fold_col = f'fold_{fold}'

    # 1. Fold마다 데이터로더 새롭게 구성 (train:4096샘플링, val:전체)
    train_ds = H5Dataset(split="train", fold_col=current_fold_col)
    val_ds = H5Dataset(split="val", fold_col=current_fold_col)
    # Val은 패치 개수가 환자마다 달라 batch_size=1 필수
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

    # 2. Fold마다 모델과 옵티마이저 완전히 새로 초기화 (데이터 누수 방지)
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
    # 불균형 데이터를 위해 Focal Loss 적용
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)

    # 최고 성능을 추적하기 위한 변수 초기화
    best_metrics = {}
    best_val_auprc = 0.0
    best_val_auroc = 0.0

    # --- 🚨 얼리 스토핑 관련 변수 추가 ---
    patience = 5  # 최고 성능을 갱신하지 못하고 5에폭을 버티면 종료
    early_stop_counter = 0
    best_thresh = 0

    # 3. Training & Validation Loop
    for epoch in range(num_epochs):
        print(f'Epoch_{epoch} start')

        accumulation_steps = 16
        model.train()
        optimizer.zero_grad()
        
        for i, (features, coords, labels) in enumerate(tqdm(train_loader)):
            features, labels = {'features': features.to(device)}, labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # 4. Validation Evaluation (해당 Fold의 검증 성능 측정)
        model.eval()
        all_labels, all_probs = [], []
        correct, total = 0, 0

        with torch.no_grad():
            for features, coords, labels in tqdm(val_loader):
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

        try:
            auroc = roc_auc_score(all_labels, all_probs)
            auprc = average_precision_score(all_labels, all_probs)
            
            # 🔥 F1-Score Maximization 적용 구간 🔥
            # thresholds 배열보다 precisions, recalls 배열의 길이가 1 더 깁니다.
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            
            # 분모가 0이 되는 것을 방지하기 위해 1e-8(epsilon) 추가
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            
            # F1 Score가 최대가 되는 인덱스 추출
            best_idx = np.argmax(f1_scores)
            thresh = thresholds[best_idx]
            
            # 극단적인 임계값(전부 다 0이거나 전부 다 1로 찍는 경우) 방지를 위한 클리핑(Clipping)
            # best_thresh = np.clip(best_thresh, 0.05, 0.95)
        except ValueError:
            auroc = 0.5
            auprc = 0.0
            thresh = 0.5

        # 찾아낸 최적의 Threshold로 예측값(preds)을 한 번에 생성
        all_preds = (all_probs >= thresh).astype(float)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # # 현재 Epoch 지표 계산
        # val_auc = roc_auc_score(all_labels, all_probs)
        # val_auprc = average_precision_score(all_labels, all_probs)
        # val_accuracy = correct / total

        # print(f"[Fold {fold} | Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val AUC: {auroc:.4f}")
        print(f'====> Epoch: {epoch:02d} | Valid AUROC: {auroc:.4f} | Valid AUPRC: {auprc:.4f} | Valid F1: {f1:.4f} | Valid Thresh: {thresh:.4f} <====')
        # 기준을 AUPRC로 변경하여 최고의 분류 성능을 가진 모델 가중치를 저장
        if auroc > best_val_auroc:
            best_val_auprc = auprc
            best_val_auroc = auroc
            best_thresh = thresh
            checkpoint_path = os.path.join(MODEL_PATH, f"best_model_fold{fold}.pt")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_thresh': best_thresh.item(),  # 👈 여기서 최적 임계값 저장
                'auprc': best_val_auprc,
                'auroc':best_val_auroc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"🔥 Fold {fold+1} 최고 성능 갱신! 모델 저장됨 (AUROC: {auroc:.4f}), (AUPRC: {auprc:.4f}), Best Threshold: {thresh:.4f}")
            # 카운터 초기화
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"{patience} epoch 동안 성능 개선이 없어 Fold {fold+1} 학습 조기 종료")
            break

    # 모든 Epoch 종료 후, 해당 Fold의 최고 성능 지표를 최종 결과에 추가
    