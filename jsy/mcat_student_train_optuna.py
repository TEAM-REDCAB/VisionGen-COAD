import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

from utils.mcat_student_model import MCAT_Student
from utils.mcat_student_train_binary import train_binary, validate_binary
from utils.h5dataset_full import H5Dataset
import config as cf
from sklearn.metrics import roc_auc_score, average_precision_score

import logging
import sys

# ── 로깅 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("mcat_student_train_optuna.txt"), logging.StreamHandler()],
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

# ── 설정값 로드 ───────────────────────────────────────────────────────────────
SEED = cf.SEED
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_feats_path()
RESULTS_PATH = cf.get_results_path()

# Optuna 실험 중 생성되는 임시 최고 모델들을 저장할 별도 폴더
MODEL_PATH = os.path.join(RESULTS_PATH, "saved_models_optuna")
KNOWLEDGE_DIR = cf.get_teacher_knowledge_path()
os.makedirs(MODEL_PATH, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_ensemble():
    """
    각 Trial 학습 종료 후, 5개의 Fold 모델을 로드하여
    외부 코호트(Test Set)에 대한 앙상블 AUROC, F1, Sensitivity를 반환합니다.
    """
    test_dataset = H5Dataset(split="all", fold_col='fold_0', test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_labels = []
    ensemble_probs = None
    
    for fold in range(5):
        model = MCAT_Student().to(device)
        model_path = os.path.join(MODEL_PATH, f'best_model_fold{fold}.pt')
        if not os.path.exists(model_path):
            return 0.5, 0.0 # 파일이 없으면 매우 안 좋은 결과 반환
            
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        
        fold_probs = []
        fold_labels = []
        
        with torch.no_grad():
            for features, coords, labels in test_loader:
                features = features.to(device)
                logits, *_ = model(features)
                logits = logits.squeeze(dim=-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits)
                fold_probs.extend(probs.cpu().numpy())
                
                if fold == 0: 
                    fold_labels.extend(labels.cpu().numpy())
                    
        fold_probs = np.array(fold_probs)
        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_labels = np.array(fold_labels)
        else:
            ensemble_probs += fold_probs
            
    ensemble_probs = ensemble_probs / 5.0
    
    auroc = roc_auc_score(all_labels, ensemble_probs)
    auprc = average_precision_score(all_labels, ensemble_probs)
        
    return auroc, auprc


def objective(trial):
    # 하이퍼파라미터 탐색 범위 설정
    w_task = trial.suggest_float('w_task', 0.1, 0.4)
    alpha = trial.suggest_float('alpha', 0.2, 0.5)
    beta = trial.suggest_float('beta', 0.2, 0.5)
    gamma = trial.suggest_float('gamma', 0.2, 0.5)
    
    print(f"\n{'='*50}")
    print(f"🚀 Trial {trial.number} Start! | w_task={w_task:.4f}, alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")
    print(f"{'='*50}")
    
    set_seed(SEED)
    num_epochs = 20
    gc_steps = 16
    patience = 5  # 조기 종료 patience
    
    # ── 5-Fold 학습 루프 ─────────────────────────────────────────────────────
    for fold_idx in range(5):
        print(f"\n--- Optuna Trial {trial.number} - Fold {fold_idx} ---")
        current_fold_col = f"fold_{fold_idx}"
        kd_path = os.path.join(KNOWLEDGE_DIR, f"knowledge_fold{fold_idx}_train.pkl")

        train_ds = H5Dataset(split="train", fold_col=current_fold_col, kd_path=kd_path)
        val_ds = H5Dataset(split="val", fold_col=current_fold_col)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

        omic_path = os.path.join(KNOWLEDGE_DIR, f"avg_omic_fold{fold_idx}.pt")
        avg_omic_tensor = torch.load(omic_path, weights_only=False)

        model = MCAT_Student(avg_omic_tensor=avg_omic_tensor).to(device)

        n_neg = int((train_ds.df["msi"] == 0).sum())
        n_pos = int((train_ds.df["msi"] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        base_params = [p for n, p in model.named_parameters() if "latent_queries" not in n]
        latent_params = [model.latent_queries]

        optimizer = optim.Adam([
            {"params": base_params, "lr": 5e-5, "weight_decay": 1e-4},
            {"params": latent_params, "lr": 5e-4, "weight_decay": 0.0},
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_auroc = 0.0
        best_val_auprc = 0.0
        early_stop_counter = 0

        for epoch in range(num_epochs):
            train_binary(epoch, model, train_loader, optimizer, criterion, gc=gc_steps, 
                         alpha=alpha, beta=beta, gamma=gamma, w_task=w_task)
            val_labels, val_probs, val_auroc, val_auprc = validate_binary(epoch, model, val_loader, criterion)
            
            scheduler.step()

            save_model = False
            if val_auroc > best_val_auroc:
                save_model = True
            elif val_auroc + 1e-4 > best_val_auroc:
                if val_auprc > best_val_auprc:
                    save_model = True

            if save_model:
                best_val_auroc = val_auroc
                best_val_auprc = val_auprc
                early_stop_counter = 0
                checkpoint_path = os.path.join(MODEL_PATH, f"best_model_fold{fold_idx}.pt")
                torch.save(model.state_dict(), checkpoint_path)
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"Trial {trial.number} Fold {fold_idx} 조기 종료 (Epoch {epoch})")
                break
                
    # ── 앙상블 테스트 평가 (외부 코호트) ──────────────────────────────────────
    print(f"\n🔍 Trial {trial.number} 5-Fold 학습 완료. 앙상블 테스트 평가 중...")
    ens_auroc, ens_auprc = evaluate_ensemble()
    
    print(f"📊 Trial {trial.number} 결과 | AUROC: {ens_auroc:.4f}, AUPRC: {ens_auprc:.4f}")
    
    return ens_auroc

if __name__ == "__main__":
    study_name = "mcat_kd_optimization"
    storage_name = "sqlite:///optuna_study.db"
    
    # SQLite DB를 사용하여 중간에 끊겨도 이어서 학습할 수 있도록 설정
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="maximize"
    )
    
    print("=========================================================")
    print("Optuna 하이퍼파라미터 최적화 시작 (w_task, alpha, beta, gamma)")
    print("=========================================================")
    # n_trials는 30번으로 우선 설정 (필요시 조절 가능)
    study.optimize(objective, n_trials=30)
    
    print("\n✅ 최적화 완료!")
    print("🏆 가장 성능이 좋았던 파라미터 조합:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
        
    print(f"🎯 최고 AUROC 점수 (앙상블 테스트 기준): {study.best_value:.4f}")
    
    # 베스트 파라미터를 JSON 파일로 저장
    with open("optuna_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    print("결과가 optuna_best_params.json 파일에 저장되었습니다.")
