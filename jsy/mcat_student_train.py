import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.mcat_student_model import MCAT_Student
from utils.mcat_student_train_binary import train_binary, validate_binary
from utils.h5dataset_full import H5Dataset
import config as cf
from utils.oof_results import oof_results
import logging
import sys
from mcat_student_test_ensemble import test_and_visualize

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


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    gc_steps = 16
    alpha = 0.4163731668137921
    beta = 0.25514803906679484
    gamma = 0.2914255011813559
    w_task = 0.3588696556112462
    # Task=0.3588696556112462 | Logit(alpha)=0.4163731668137921 | Attn(beta)=0.25514803906679484 | Path(gamma)=0.2914255011813559
    oof_probs = None
    oof_labels = None

    for fold_idx in range(5):
        print(
            f"\n{'=' * 20} Starting Fold {fold_idx} (Knowledge Distillation) {'=' * 20}"
        )
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

        # ── 손실 함수 : BCEWithLogitsLoss + pos_weight ──────────────────────────────
        # pos_weight = MSS수 / MSI수 ≈ 3.25 → 클래스 불균형 보정
        # Focal Loss 대비 hard-example focusing 없이 부드럽게 불균형 처리 → 일반화에 유리
        # 교사 모델과 동일한 BCE 계열로 지식 전수 일관성 확보
        n_neg = int((train_ds.df["msi"] == 0).sum())
        n_pos = int((train_ds.df["msi"] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(
            device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── 옵티마이저 : latent_queries 별도 lr ──────────────────────────────────
        # latent_queries는 유전체 정보 없이 이미지로만 학습되는 핵심 파라미터.
        # 그래디언트 신호가 약하므로 lr을 10배 높이고 weight_decay를 제거.
        base_params = [
            p for n, p in model.named_parameters() if "latent_queries" not in n
        ]
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
        early_stop_counter = 0
        patience = 5

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
            val_labels, val_probs, val_auroc, val_auprc = validate_binary(
                epoch, model, val_loader, criterion
            )

            scheduler.step()

            # 기준을 AUROC로 변경하여 최고의 분류 성능을 가진 모델 가중치를 저장
            save_model = False
            # AUROC가 더 좋으면 저장, 아니면 AUPRC로 비교하여 저장
            if val_auroc > best_val_auroc:
                save_model = True
            elif val_auroc + 1e-4 > best_val_auroc:
                if val_auprc > best_val_auprc:
                    save_model = True
            if save_model:
                best_val_auroc = val_auroc
                best_val_auprc = val_auprc
                best_val_probs = val_probs.copy()
                best_val_labels = val_labels.copy()
                checkpoint_path = os.path.join(
                    MODEL_PATH, f"best_model_fold{fold_idx}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(
                    f"🔥 Fold {fold_idx} 최고 성능 갱신! 모델 저장됨 (AUROC: {val_auroc:.4f}), (AUPRC: {val_auprc:.4f})"
                )
                # 카운터 초기화
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(
                    f"{patience} epoch 동안 성능 개선이 없어 Fold {fold_idx} 학습 조기 종료"
                )
                break

        if oof_probs is None:
            oof_probs = best_val_probs.copy()
            oof_labels = best_val_labels.copy()
        else:
            oof_probs = np.concatenate([oof_probs, best_val_probs], axis=0)
            oof_labels = np.concatenate([oof_labels, best_val_labels], axis=0)
        print(
            f"Fold {fold_idx} 종료. AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}"
        )

    best_thresh = oof_results(oof_probs, oof_labels, RESULTS_PATH, "MCAT_Student_KD")
    test_and_visualize(best_thresh)


if __name__ == "__main__":
    main()
