import os
import csv
import copy
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    'save_dir': './results_teacher_v2',
}
# /home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2
# /home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath


# 💡 [안전장치] 저장 폴더가 이미 있으면 _01, _02 ... 자동 증가
_base_dir = PATHS['save_dir']
if os.path.exists(_base_dir) and len(os.listdir(_base_dir)) > 0:
    _counter = 1
    while True:
        _candidate = f"{_base_dir}_{_counter:02d}"
        if not os.path.exists(_candidate) or len(os.listdir(_candidate)) == 0:
            PATHS['save_dir'] = _candidate
            print(f"💡 [자동 저장 경로 변경] '{_base_dir}' 이미 존재 → '{PATHS['save_dir']}' 로 자동 변경")
            break
        _counter += 1
os.makedirs(PATHS['save_dir'], exist_ok=True)



# =========================================================================
# 하이퍼파라미터 (MCAT 원 논문 기준)
# =========================================================================
CFG = {
    'n_folds':       5,
    'max_epochs':    20,
    'lr':            2e-4,
    'weight_decay':  1e-5,
    'es_patience':   10,
    'es_stop_epoch': 20,
    'seed':          42,
    'holdout_ratio': 0.2,   # 💡 Final Test Set 비율
}


# =========================================================================
# EarlyStopping (cyl 코드 그대로)
# =========================================================================
class EarlyStopping:
    def __init__(self, patience=10, stop_epoch=20, verbose=True):
        self.patience    = patience
        self.stop_epoch  = stop_epoch
        self.verbose     = verbose
        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False
        self.val_loss_min = np.inf
        self.best_state  = None

    def __call__(self, epoch, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"  [ES] {self.counter}/{self.patience} "
                      f"(val_loss 미개선: {val_loss:.4f} >= best {self.val_loss_min:.4f})")
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score   = score
            self.val_loss_min = val_loss
            self.best_state   = copy.deepcopy(model.state_dict())
            self.counter      = 0


# =========================================================================
# XAI 기록 (cyl 코드 기반, 함수로 분리)
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

    A_coattn_2d  = resolve_coattn_shape(attn['coattn'], n_omic=1425)
    omic_imp     = A_coattn_2d.sum(dim=1)
    top_omic     = omic_imp.argmax().item()

    patch_row    = A_coattn_2d[top_omic]
    top_patch    = patch_row.argmax().item()
    attn_score   = patch_row[top_patch].item()

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
    N = len(dataset)
    print(f"[데이터] 전체 환자: {N}명\n")

    # 환자별 정답 라벨 전체 추출
    all_labels  = [dataset.slide_data['label'][i] for i in range(N)]
    all_indices = list(range(N))

    # =========================================================================
    # 💡 [v22 추가] Step 0: 전체에서 Final Test Set 20% 먼저 홀드아웃
    #   - 이 53명(약)은 5-Fold CV 어디에도 절대 사용하지 않음
    #   - Stratified 분리로 MSI/MSS 비율 동일하게 유지
    # =========================================================================
    cv_indices, final_test_indices = train_test_split(
        all_indices,
        test_size=CFG['holdout_ratio'],
        random_state=CFG['seed'],
        shuffle=True,
        stratify=all_labels,
    )
    cv_labels         = [all_labels[i] for i in cv_indices]
    final_test_labels = [all_labels[i] for i in final_test_indices]

    n_msi_cv = sum(cv_labels)
    n_msi_ft = sum(final_test_labels)
    print(f"{'='*55}")
    print(f" [데이터 분할]")
    print(f"  5-Fold CV용  : {len(cv_indices)}명 (MSIMUT {n_msi_cv} | MSS {len(cv_indices)-n_msi_cv})")
    print(f"  Final Test   : {len(final_test_indices)}명 (MSIMUT {n_msi_ft} | MSS {len(final_test_indices)-n_msi_ft})")
    print(f"{'='*55}\n")

    # Final Test Loader — 5-Fold 모두 완료 후 딱 1번만 사용
    final_test_loader = DataLoader(
        Subset(dataset, final_test_indices), batch_size=1, shuffle=False
    )

    # =========================================================================
    # Step 1~5: 기존 5-Fold CV 구조 그대로 (cv_indices 80% 안에서만 수행)
    #   각 Fold 내 trainval → train/val/test 분리 구조 변경 없음
    # =========================================================================
    epoch_log = []
    pred_log  = []
    fold_log  = []

    fold_csv = os.path.join(PATHS['save_dir'], 'fold_test_summary.csv')
    with open(fold_csv, 'w', newline='', encoding='utf-8-sig') as f:
        csv.DictWriter(f, fieldnames=['fold', 'test_loss', 'test_acc', 'test_auc', 'stopped_epoch']).writeheader()

    skf     = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
    scaler  = GradScaler(device='cuda')
    loss_fn = nn.CrossEntropyLoss()

    best_fold_auc   = -1.0
    best_fold_num   = -1
    best_fold_state = None  # Final Evaluation에 쓸 Best Fold 가중치

    # StratifiedKFold: cv_indices(80%) 안에서만 수행
    for fold, (trainval_local, test_local) in enumerate(
        skf.split(range(len(cv_indices)), cv_labels)
    ):
        # local → global 인덱스 변환
        trainval_idx = [cv_indices[i] for i in trainval_local]
        test_idx     = [cv_indices[i] for i in test_local]

        print(f"\n{'='*55}")
        print(f" [Fold {fold+1}/{CFG['n_folds']}] 학습 시작")
        print(f"{'='*55}")

        model = MCATTeacher(
            vocab_sizes=vocab_sizes, path_dim=1536, n_classes=2, mcat_path=PATHS['mcat']
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']
        )

        # Train / Val 분할 (stratify 유지)
        trainval_labels = [all_labels[i] for i in trainval_idx]
        train_idx, val_idx = train_test_split(
            list(trainval_idx), test_size=0.2, random_state=CFG['seed'],
            shuffle=True, stratify=trainval_labels
        )

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=1, shuffle=False)
        test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=1, shuffle=False)

        print(f"[분할] Train {len(train_idx)} | Val {len(val_idx)} | Test(Fold 내) {len(test_idx)}")

        es = EarlyStopping(patience=CFG['es_patience'], stop_epoch=CFG['es_stop_epoch'])
        stopped_epoch = CFG['max_epochs']

        for epoch in range(CFG['max_epochs']):
            tr_loss, tr_acc, tr_auc, _ = run_epoch(
                model, train_loader, loss_fn, device,
                optimizer=optimizer, is_train=True, scaler=scaler
            )
            val_loss, val_acc, val_auc, _ = run_epoch(
                model, val_loader, loss_fn, device, is_train=False
            )

            epoch_log.append({
                'fold': fold+1, 'epoch': epoch+1,
                'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc*100, 2), 'train_auc': round(tr_auc, 4),
                'val_loss':   round(val_loss, 4), 'val_acc':   round(val_acc*100, 2), 'val_auc':   round(val_auc, 4),
            })

            print(f"  [Fold {fold+1} | Epoch {epoch+1:02d}/{CFG['max_epochs']}] "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc*100:.2f}% AUC {tr_auc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% AUC {val_auc:.4f}")

            es(epoch, val_loss, model)
            if es.early_stop:
                print(f"  [Early Stop] Fold {fold+1} Epoch {epoch+1}")
                stopped_epoch = epoch + 1 - es.counter
                break
            if not es.early_stop:
                stopped_epoch = CFG['max_epochs']

        if es.best_state:
            model.load_state_dict(es.best_state)
            print(f"  [복원] Best val_loss={es.val_loss_min:.4f}")

        ckpt_path = os.path.join(PATHS['save_dir'], f'teacher_fold{fold+1}.pth')
        torch.save(model.state_dict(), ckpt_path)

        # Fold 내 Test 평가 (Finding Parameters — 기존 그대로)
        test_loss, test_acc, test_auc, test_results = run_epoch(
            model, test_loader, loss_fn, device, is_train=False, save_xai=True
        )
        print(f"\n  [Fold {fold+1} Test] Loss {test_loss:.4f} | Acc {test_acc*100:.2f}% | AUC {test_auc:.4f}")

        for r in test_results:
            rec = build_xai_record(r, var_vocab_inv, vc_vocab_inv, func_vocab_inv, fold+1)
            rec.update({'test_loss': round(test_loss, 4), 'test_acc': round(test_acc*100, 2), 'test_auc': round(test_auc, 4)})
            pred_log.append(rec)

        new_row = {
            'fold': fold+1, 'test_loss': round(test_loss, 4),
            'test_acc': round(test_acc*100, 2), 'test_auc': round(test_auc, 4),
            'stopped_epoch': stopped_epoch,
        }
        fold_log.append(new_row)
        with open(fold_csv, 'a', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=['fold', 'test_loss', 'test_acc', 'test_auc', 'stopped_epoch']).writerow(new_row)

        # Best Fold 추적 (Final Evaluation에 쓸 모델)
        if test_auc > best_fold_auc:
            best_fold_auc   = test_auc
            best_fold_num   = fold + 1
            best_fold_state = copy.deepcopy(model.state_dict())
            print(f"  ⭐ [Best Fold 갱신] Fold {fold+1} Test AUC {test_auc:.4f}")

        del model, optimizer
        torch.cuda.empty_cache()

    # =========================================================================
    # 💡 Final Evaluation: Best Fold 모델로 홀드아웃 53명 딱 1번 평가
    # =========================================================================
    print(f"\n{'='*55}")
    print(f" [Final Evaluation] Best Fold {best_fold_num} 모델 사용")
    print(f" Hold-out된 Test set {len(final_test_indices)}명 최종 평가 시작...")
    print(f"{'='*55}")

    final_model = MCATTeacher(
        vocab_sizes=vocab_sizes, path_dim=1536, n_classes=2, mcat_path=PATHS['mcat']
    ).to(device)
    final_model.load_state_dict(best_fold_state)

    final_loss, final_acc, final_auc, final_results = run_epoch(
        final_model, final_test_loader, loss_fn, device, is_train=False, save_xai=True
    )
    print(f"\n  [Final Test] Loss {final_loss:.4f} | Acc {final_acc*100:.2f}% | AUC {final_auc:.4f}")

    # =========================================================================
    # 최종 저장
    # =========================================================================
    epoch_csv = os.path.join(PATHS['save_dir'], 'epoch_summary.csv')
    with open(epoch_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_log[0].keys())
        writer.writeheader(); writer.writerows(epoch_log)

    # 5-Fold 내 Fold별 test XAI
    pred_csv = os.path.join(PATHS['save_dir'], 'MSI_MSS_prediction.csv')
    with open(pred_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=pred_log[0].keys())
        writer.writeheader(); writer.writerows(pred_log)

    # Final Evaluation XAI
    final_xai = []
    for r in final_results:
        rec = build_xai_record(r, var_vocab_inv, vc_vocab_inv, func_vocab_inv, fold=best_fold_num)
        rec.update({
            'final_loss': round(final_loss, 4),
            'final_acc':  round(final_acc*100, 2),
            'final_auc':  round(final_auc, 4),
        })
        final_xai.append(rec)

    final_csv = os.path.join(PATHS['save_dir'], 'MSI_MSS_prediction_final.csv')
    with open(final_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=final_xai[0].keys())
        writer.writeheader(); writer.writerows(final_xai)

    # 최종 요약 출력
    print(f"\n{'='*55}")
    print(f" [5-Fold CV 요약 (Finding Parameters)]")
    print(f"  Fold Test AUC 평균: {sum(r['test_auc'] for r in fold_log)/len(fold_log):.4f}")
    print(f"  Fold Test Acc 평균: {sum(r['test_acc'] for r in fold_log)/len(fold_log):.2f}%")
    print(f"\n [Final Evaluation (Hold-out {len(final_test_indices)}명)]")
    print(f"  AUC: {final_auc:.4f} | Acc: {final_acc*100:.2f}%  (Best Fold: {best_fold_num})")
    print(f"{'='*55}")
    print("[완료] Teacher Hold-out + 5-Fold CV 학습 완료!")


if __name__ == '__main__':
    main()
