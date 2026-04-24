"""
main_teacher.py
Teacher 모델 5-Fold CV 학습 진입점 (MCAT 원 논문 방식)

분할 전략:
    - StratifiedKFold: train 80% / val 20%
    - 별도 test set 없음
    - 5개 fold val 성능 평균 = 최종 보고 성능
    - val이 early stopping + 성능 평가를 동시에 담당

실행:
    python main_teacher.py
"""

import os
import csv
import copy
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from dataset import Pathomic_Classification_Dataset
from models.mcat_teacher import MCATTeacher
from train import run_epoch

# =========================================================================
# 경로 설정  ← 환경에 맞게 수정
# =========================================================================
PATHS = {
    'clin_csv': './data/tcga_coad_all_clean.csv',
    'mut_csv':  './data/genomic/preprocessed_mutation_data.csv',
    'npy':      './data/genomic/genomic_input_matrix.npy',
    'pkl':      './data/genomic/genomic_encoding_states.pkl',
    'wsi_dir':  '/home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath',
    'mcat':     './mcat',
    'save_dir': './results_teacher',
}

# =========================================================================
# 하이퍼파라미터 (MCAT 원 논문 기준)
# =========================================================================
CFG = {
    'n_folds':       5,
    'max_epochs':    20,
    'lr':            2e-4,
    'weight_decay':  1e-5,
    'es_patience':   10,
    'es_stop_epoch': 5,   # max_epochs보다 충분히 작아야 early stopping이 실제로 발동됨
    'seed':          42,
}


# =========================================================================
# EarlyStopping
# =========================================================================
class EarlyStopping:
    def __init__(self, patience=10, stop_epoch=5, verbose=True):
        self.patience     = patience
        self.stop_epoch   = stop_epoch   # 이 epoch 이후부터 early stop 허용
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.inf
        self.best_state   = None

    def __call__(self, epoch, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self._update_best(score, val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"  [ES] {self.counter}/{self.patience} "
                      f"(val_loss 미개선: {val_loss:.4f} >= best {self.val_loss_min:.4f})")
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self._update_best(score, val_loss, model)

    def _update_best(self, score, val_loss, model):
        self.best_score   = score
        self.val_loss_min = val_loss
        self.best_state   = copy.deepcopy(model.state_dict())
        self.counter      = 0


# =========================================================================
# XAI 기록
# =========================================================================
def resolve_coattn_shape(A_coattn, n_omic=1425):
    for dim_idx in range(A_coattn.dim()):
        if A_coattn.shape[dim_idx] == n_omic:
            A_coattn = A_coattn.transpose(0, dim_idx)
            return A_coattn.reshape(n_omic, -1)
    return A_coattn.reshape(1, -1)


def build_xai_record(result, var_vocab_inv, vc_vocab_inv, func_vocab_inv, fold):
    attn   = result['attn_scores']
    g_feat = result['genomic_features']
    pred   = result['pred_class']
    prob   = result['msi_prob']
    true   = result['true_label']

    pred_str = 'MSIMUT' if pred == 1 else 'MSS'
    true_str = 'MSIMUT' if true == 1 else 'MSS'
    prob_pct = round(prob * 100 if pred == 1 else (1 - prob) * 100, 1)

    A_coattn_2d = resolve_coattn_shape(attn['coattn'], n_omic=1425)
    omic_imp    = A_coattn_2d.sum(dim=1)
    top_omic    = omic_imp.argmax().item()

    patch_row   = A_coattn_2d[top_omic]
    top_patch   = patch_row.argmax().item()
    attn_score  = patch_row[top_patch].item()

    row      = g_feat[top_omic]
    var_name = var_vocab_inv.get(int(row[0].item()), f'Var#{int(row[0].item())}')
    vc_name  = vc_vocab_inv.get(int(row[1].item()),  f'VC#{int(row[1].item())}')
    funcs    = [func_vocab_inv[int(f.item())] for f in row[2:8].long()
                if int(f.item()) > 0 and int(f.item()) in func_vocab_inv]
    func_str = ', '.join(funcs) if funcs else 'N/A'

    return {
        'fold': fold, 'patient': result['patient_id'],
        'true_label': true_str, 'prediction': pred_str, 'probability(%)': prob_pct,
        'top_variant': var_name, 'variant_class': vc_name, 'gene_functions': func_str,
        'top_patch_idx': top_patch, 'attention_score': round(attn_score, 4),
    }


# =========================================================================
# main
# =========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[시스템] device: {device}\n")

    os.makedirs(PATHS['save_dir'], exist_ok=True)

    # 0. 단어 사전 로드
    with open(PATHS['pkl'], 'rb') as f:
        enc = pickle.load(f)
    vocab_sizes = {
        'var':  len(enc['var_vocab']),
        'vc':   len(enc['vc_vocab']),
        'func': len(enc['func_vocab']),
    }
    var_vocab_inv  = {v: k for k, v in enc['var_vocab'].items()}
    vc_vocab_inv   = {v: k for k, v in enc['vc_vocab'].items()}
    func_vocab_inv = {v: k for k, v in enc['func_vocab'].items()}

    # 1. 전체 데이터셋
    dataset = Pathomic_Classification_Dataset(
        clin_csv_path=PATHS['clin_csv'],
        mut_csv_path=PATHS['mut_csv'],
        npy_path=PATHS['npy'],
        data_dir=PATHS['wsi_dir'],
    )
    print(f"[데이터] 전체 환자: {len(dataset)}명\n")

    all_labels = [dataset.slide_data['label'][i] for i in range(len(dataset))]

    # 2. 로그 저장소
    epoch_log = []
    pred_log  = []
    fold_log  = []

    # fold별 val 성능 요약 CSV (헤더 미리 작성)
    fold_csv = os.path.join(PATHS['save_dir'], 'fold_val_summary.csv')
    fold_fieldnames = ['fold', 'val_loss', 'val_acc', 'val_auc', 'stopped_epoch']
    with open(fold_csv, 'w', newline='', encoding='utf-8-sig') as f:
        csv.DictWriter(f, fieldnames=fold_fieldnames).writeheader()

    # 3. 5-Fold CV
    # StratifiedKFold가 직접 train(80%) / val(20%)로 분할
    # → train_test_split 불필요, 모든 환자가 정확히 한 번 val에 들어감
    skf     = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
    scaler  = GradScaler(device='cuda')
    loss_fn = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), all_labels)):
        print(f"\n{'='*55}")
        print(f" [Fold {fold+1}/{CFG['n_folds']}] 학습 시작")
        print(f"{'='*55}")
        print(f"[분할] Train {len(train_idx)} | Val {len(val_idx)}")

        model = MCATTeacher(
            vocab_sizes=vocab_sizes, path_dim=1536, n_classes=2, mcat_path=PATHS['mcat']
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']
        )

        train_loader = DataLoader(Subset(dataset, list(train_idx)), batch_size=1, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, list(val_idx)),   batch_size=1, shuffle=False)

        es = EarlyStopping(patience=CFG['es_patience'], stop_epoch=CFG['es_stop_epoch'])
        stopped_epoch = CFG['max_epochs']  # early stop 없으면 max_epochs로 기록

        for epoch in range(CFG['max_epochs']):
            tr_loss, tr_acc, tr_auc, _ = run_epoch(
                model, train_loader, loss_fn, device,
                optimizer=optimizer, is_train=True, scaler=scaler
            )
            val_loss, val_acc, val_auc, _ = run_epoch(
                model, val_loader, loss_fn, device, is_train=False
            )

            epoch_log.append({
                'fold':       fold + 1,
                'epoch':      epoch + 1,
                'train_loss': round(tr_loss,       4),
                'train_acc':  round(tr_acc * 100,  2),
                'train_auc':  round(tr_auc,        4),
                'val_loss':   round(val_loss,       4),
                'val_acc':    round(val_acc * 100,  2),
                'val_auc':    round(val_auc,        4),
            })

            print(f"  [Fold {fold+1} | Epoch {epoch+1:02d}/{CFG['max_epochs']}] "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc*100:.2f}% AUC {tr_auc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% AUC {val_auc:.4f}")

            es(epoch, val_loss, model)
            if es.early_stop:
                stopped_epoch = epoch + 1 - es.counter
                print(f"  [Early Stop] Fold {fold+1} | best epoch: {stopped_epoch}")
                break

        # best checkpoint 복원
        if es.best_state:
            model.load_state_dict(es.best_state)
            print(f"  [복원] Best val_loss={es.val_loss_min:.4f}")

        # 체크포인트 저장
        ckpt_path = os.path.join(PATHS['save_dir'], f'teacher_fold{fold+1}.pth')
        torch.save(model.state_dict(), ckpt_path)

        # ── best 모델로 val 재평가 (XAI 포함) ──────────────────────────
        # early stopping으로 best state 복원 후 val을 다시 돌려
        # 최종 보고 성능(best epoch 기준)을 기록합니다.
        best_val_loss, best_val_acc, best_val_auc, val_results = run_epoch(
            model, val_loader, loss_fn, device, is_train=False, save_xai=True
        )
        print(f"\n  [Fold {fold+1} Val (best ckpt)] "
              f"Loss {best_val_loss:.4f} | Acc {best_val_acc*100:.2f}% | AUC {best_val_auc:.4f}")

        # XAI 기록
        for r in val_results:
            rec = build_xai_record(r, var_vocab_inv, vc_vocab_inv, func_vocab_inv, fold + 1)
            rec.update({
                'val_loss': round(best_val_loss, 4),
                'val_acc':  round(best_val_acc * 100, 2),
                'val_auc':  round(best_val_auc, 4),
            })
            pred_log.append(rec)

        # fold 요약
        new_row = {
            'fold':          fold + 1,
            'val_loss':      round(best_val_loss,       4),
            'val_acc':       round(best_val_acc * 100,  2),
            'val_auc':       round(best_val_auc,        4),
            'stopped_epoch': stopped_epoch,
        }
        fold_log.append(new_row)
        with open(fold_csv, 'a', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=fold_fieldnames).writerow(new_row)

        del model, optimizer
        torch.cuda.empty_cache()

    # =========================================================================
    # 4. 최종 저장
    # =========================================================================
    # 에폭별 로그
    epoch_csv = os.path.join(PATHS['save_dir'], 'epoch_summary.csv')
    with open(epoch_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_log)

    # 환자별 XAI 예측
    if pred_log:
        pred_csv = os.path.join(PATHS['save_dir'], 'MSI_MSS_prediction.csv')
        with open(pred_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=pred_log[0].keys())
            writer.writeheader()
            writer.writerows(pred_log)

    # 5-Fold 평균 출력
    mean_auc = sum(r['val_auc'] for r in fold_log) / len(fold_log)
    mean_acc = sum(r['val_acc'] for r in fold_log) / len(fold_log)
    print(f"\n{'='*55}")
    print(f" [5-Fold 최종 요약]")
    for r in fold_log:
        print(f"  Fold {r['fold']} | Val AUC {r['val_auc']:.4f} | Acc {r['val_acc']:.2f}% "
              f"| stopped epoch {r['stopped_epoch']}")
    print(f"  {'─'*45}")
    print(f"  평균 Val AUC: {mean_auc:.4f}")
    print(f"  평균 Val Acc: {mean_acc:.2f}%")
    print(f"{'='*55}")
    print("[완료] Teacher 5-Fold CV 학습 완료!")


if __name__ == '__main__':
    main()