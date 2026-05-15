from pandas.core import window
from pandas.core import window
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.mcat_student_model import MCAT_Student, BinaryFocalLoss
from utils.mcat_student_train_binary import train_binary, validate_binary
from utils.h5dataset_full import H5Dataset
import config as cf

import logging
import sys

# ── 로깅 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("mcat_student_train.txt"), logging.StreamHandler()],
)


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message.strip())

    def flush(self):
        pass


sys.stdout = LoggerWriter(logging.info)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_lr", type=float, default=5e-5)
parser.add_argument("--latent_lr", type=float, default=5e-4)
parser.add_argument("--focal_alpha", type=float, default=0.75)
parser.add_argument("--focal_gamma", type=float, default=2.0)
args = parser.parse_args()

# ── 설정값 로드 ───────────────────────────────────────────────────────────────
SEED = cf.SEED
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_feats_path()
RESULTS_PATH = cf.get_results_path()
MODEL_PATH = os.path.join(RESULTS_PATH, "saved_models_mcat_kd")
KNOWLEDGE_DIR = cf.get_teacher_knowledge_path()

os.makedirs(MODEL_PATH, exist_ok=True)


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
alpha = 0.15
beta = 0.5
gamma = 0.35
w_task = 0.2


fold_results = []

for fold_idx in range(5):
    print(f"\n{'=' * 20} Starting Fold {fold_idx} (Knowledge Distillation) {'=' * 20}")
    current_fold_col = f"fold_{fold_idx}"

    kd_path = os.path.join(KNOWLEDGE_DIR, f"knowledge_fold{fold_idx}_train.pkl")

    # ── 데이터로더 ────────────────────────────────────────────────────────────
    train_ds = H5Dataset(split="train", fold_col=current_fold_col, kd_path=kd_path)
    val_ds = H5Dataset(split="val", fold_col=current_fold_col)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(SEED),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        worker_init_fn=lambda _: np.random.seed(SEED),
    )

    # ── 모델 초기화 ───────────────────────────────────────────────────────────
    omic_path = os.path.join(KNOWLEDGE_DIR, f"avg_omic_fold{fold_idx}.pt")
    avg_omic_tensor = torch.load(omic_path, weights_only=False)

    model = MCAT_Student(avg_omic_tensor=avg_omic_tensor).to(device)

    # ── 손실 함수 : BinaryFocalLoss ───────────────────────────────────────────
    # alpha=0.75 → MSI(양성) 클래스에 더 높은 가중치 부여
    # 클래스 불균형(MSI < MSS)으로 모델이 MSS로 붕괴하는 현상 방지
    criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(
        device
    )
    # criterion = nn.BCEWithLogitsLoss()

    # ── 옵티마이저 : latent_queries 별도 lr ──────────────────────────────────
    # latent_queries는 유전체 정보 없이 이미지로만 학습되는 핵심 파라미터.
    # 그래디언트 신호가 약하므로 lr을 10배 높이고 weight_decay를 제거.
    base_params = [p for n, p in model.named_parameters() if "latent_queries" not in n]
    latent_params = [model.latent_queries]

    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": args.base_lr, "weight_decay": 1e-4},
            {
                "params": latent_params,
                "lr": args.latent_lr,
                "weight_decay": 0.0,
            },  # 10x lr, no decay
        ]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_auroc = 0.0
    best_val_auprc = 0.0
    best_thresh = 0.5

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        print(f"Fold_{fold_idx} Epoch_{epoch} start")
        train_binary(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            gc=gc_steps,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_task=w_task,
        )
        val_auroc, val_auprc, val_thresh = validate_binary(
            epoch, model, val_loader, criterion
        )

        scheduler.step()

        # 기준을 AUROC로 변경
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_auprc = val_auprc
            best_thresh = val_thresh

            checkpoint_path = os.path.join(MODEL_PATH, f"best_model_fold{fold_idx}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_thresh": val_thresh,
                    "auroc": val_auroc,
                    "auprc": val_auprc,
                },
                checkpoint_path,
            )
            print(
                f"🔥 Fold {fold_idx} 최고 성능 갱신! "
                f"(AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}, Thresh: {val_thresh:.4f})"
            )

    fold_results.append(best_val_auroc)
    print(
        f"Fold {fold_idx} 종료. "
        f"AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}, Thresh: {best_thresh:.4f}"
    )

# ── 최종 결과 ─────────────────────────────────────────────────────────────────
print("\n================ 최종 결과 ================")
for i, auroc in enumerate(fold_results):
    print(f"Fold {i} Best AUROC: {auroc:.4f}")
print(f"5-Fold 평균 AUROC: {np.mean(fold_results):.4f}")

import mcat_student_test

mcat_student_test.test_and_visualize()
import mcat_student_test_ensemble

mcat_student_test_ensemble.test_and_visualize()
