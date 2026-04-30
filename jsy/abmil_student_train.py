import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.abmil_model import BinaryClassificationModel
from utils.mcat_student_train_binary import train_binary, validate_binary
from utils.h5dataset_full import H5Dataset # kd_path를 처리할 수 있게 수정된 데이터셋 사용
from utils.binary_focal_loss import BinaryFocalLoss
import config as cf

import logging
import sys

# 1. 로그 설정 (시간, 로그 레벨, 메시지 형식 지정)
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s [%(levelname)s] %(message)s',
    format='%(message)s',
    handlers=[
        logging.FileHandler("abmil_student_train.txt"), # 파일 저장
        logging.StreamHandler() # 콘솔에도 동시에 출력
    ]
)

# 2. print 문을 logging으로 리다이렉트하는 클래스
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip(): # 빈 줄이 아닐 때만 기록
            self.level(message.strip())

    def flush(self):
        pass

# 3. 시스템의 표준 출력(stdout)과 에러(stderr)를 logging에 연결
sys.stdout = LoggerWriter(logging.info)
# sys.stderr = LoggerWriter(logging.error)

# 설정값 로드
SEED = cf.SEED
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_feats_path()
RESULTS_PATH = cf.get_results_path()
MODEL_PATH = os.path.join(RESULTS_PATH, 'saved_models_abmil_kd') # KD 전용 폴더로 분리 권장
KNOWLEDGE_DIR = cf.get_teacher_knowledge_path()

os.makedirs(MODEL_PATH, exist_ok=True)

# 시드 고정
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
gc_steps = 16

fold_results = []

for fold_idx in range(5):
    print(f"\n{'='*20} Starting Fold {fold_idx} (Knowledge Distillation) {'='*20}")
    current_fold_col = f'fold_{fold_idx}'
    
    # 💡 티처의 지식 파일 경로 설정
    kd_path = None
    kd_path = os.path.join(KNOWLEDGE_DIR, f'knowledge_fold{fold_idx}_train.pkl')

    # 1. Fold마다 데이터로더 새롭게 구성
    # Train에는 티처의 지식을 함께 로드하고, Val은 평가만 하므로 로드하지 않음
    train_ds = H5Dataset(split="train", fold_col=current_fold_col, kd_path=kd_path)
    val_ds = H5Dataset(split="val", fold_col=current_fold_col, kd_path=None)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

    # # 2. 모델 및 최적화 도구 초기화
    # omic_path = os.path.join(KNOWLEDGE_DIR, f'avg_omic_fold{fold_idx}.pt')
    # avg_omic_tensor = torch.load(omic_path)
    
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
    # criterion = BinaryFocalLoss(alpha=0.75, gamma=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
    

    best_val_auprc = 0.0
    best_val_auroc = 0.0
    best_thresh = 0
    patience = 10
    early_stop_counter = 0


    # 3. Training & Validation Loop
    for epoch in range(num_epochs):
        print(f'Epoch_{epoch} start')
        train_binary(epoch, model, train_loader, optimizer, criterion, gc=gc_steps, kd_path=kd_path)
        val_auroc, val_auprc,val_thresh = validate_binary(epoch, model, val_loader, criterion)


        # 기준을 AUROC로 변경하여 최고의 분류 성능을 가진 모델 가중치를 저장
        if val_auroc > best_val_auroc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_thresh = val_thresh
            checkpoint_path = os.path.join(MODEL_PATH, f"best_model_fold{fold_idx}.pt")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_thresh': val_thresh,  # 👈 여기서 최적 임계값 저장
                'auprc': val_auprc,
                'auroc':val_auroc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"🔥 Fold {fold_idx} 최고 성능 갱신! 모델 저장됨 (AUROC: {val_auroc:.4f}), (AUPRC: {val_auprc:.4f}), (Best Threshold: {val_thresh:.4f})")
            # 카운터 초기화
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"{patience} epoch 동안 성능 개선이 없어 Fold {fold_idx} 학습 조기 종료")
            break
            
    fold_results.append(best_val_auroc)
    print(f"Fold {fold_idx} 종료. AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}, 최적 Threshold: {best_thresh:.4f}")

# 최종 결과 출력
print("\n================ 최종 결과 ================")
for i, auroc in enumerate(fold_results):
    print(f"Fold {i+1} Best AUROC: {auroc:.4f}")
print(f"5-Fold 평균 AUROC: {np.mean(fold_results):.4f}")