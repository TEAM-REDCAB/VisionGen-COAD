import os
import csv
import copy
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
 
from dataset import Pathomic_Classification_Dataset
from model_2 import MCAT_Single_Branch_Model
 
# =========================================================================
# [MCAT 원 논문 재현] main_val_v5_fixed.py
#   - Optimizer   : Adam, lr=2e-4, weight_decay=1e-5 (원 논문 default)
#   - 5-Fold CV   : test=20%, train=64%, val=16%
#   - EarlyStopping: patience=10, stop_epoch=20
#   - AMP         : CUDA 있을 때만 활성화 (CPU fallback 추가)
#   - A_coattn    : shape 자동 감지 및 안전 처리 (핵심 에러 수정)
# =========================================================================
 
# [AMP] CUDA 여부에 따라 조건부 import
USE_AMP = torch.cuda.is_available()
if USE_AMP:
    from torch.amp import autocast, GradScaler
 
 
# =========================================================================
# [EarlyStopping 클래스] MCAT 원 논문 core_utils.py 구조 그대로 재현
# =========================================================================
class EarlyStopping:
    """val_loss가 stop_epoch 이후 patience 에폭 동안 개선 없으면 학습 중단."""
    def __init__(self, warmup=0, patience=10, stop_epoch=20, verbose=True):
        self.warmup       = warmup
        self.patience     = patience
        self.stop_epoch   = stop_epoch
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.inf
        self.best_state   = None
 
    def __call__(self, epoch, val_loss, model):
        score = -val_loss
 
        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score   = score
            self.val_loss_min = val_loss
            self.best_state   = copy.deepcopy(model.state_dict())
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] patience {self.counter}/{self.patience} "
                      f"(val_loss 미개선: {val_loss:.4f} >= best {self.val_loss_min:.4f})")
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score   = score
            self.val_loss_min = val_loss
            self.best_state   = copy.deepcopy(model.state_dict())
            self.counter      = 0
 
 
# =========================================================================
# [공통 함수] Train 또는 Eval 한 바퀴
# =========================================================================
def run_epoch(model, loader, loss_fn, device, optimizer=None, is_train=True, scaler=None, save_xai=False):
    if is_train:
        model.train()
    else:
        model.eval()
 
    total_loss = 0.0
    y_true_list, y_score_list, y_pred_list = [], [], []
    batch_results = []
 
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            path_features, genomic_features, label, patient_id = batch
            path_features    = path_features.squeeze(0).to(device)
            genomic_features = genomic_features.squeeze(0).to(device)
            label            = label.to(device)
 
            if is_train:
                optimizer.zero_grad(set_to_none=True)
 
            # ✅ [AMP] CUDA 있을 때만 FP16, 없으면 일반 FP32
            if USE_AMP:
                with autocast(device_type='cuda'):
                    logits, Y_hat, attn_scores = model(path_features, genomic_features)
                    loss = loss_fn(logits, label)
            else:
                logits, Y_hat, attn_scores = model(path_features, genomic_features)
                loss = loss_fn(logits, label)
 
            total_loss += loss.item()
 
            if is_train:
                if USE_AMP and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
 
            prob       = torch.softmax(logits.detach().float(), dim=1)
            msi_prob   = prob[0, 1].item()
            pred_class = Y_hat.item()
 
            y_true_list.append(label.item())
            y_score_list.append(msi_prob)
            y_pred_list.append(pred_class)
 
            if save_xai:
                batch_results.append({
                    'patient_id':       patient_id[0],
                    'attn_scores':      {k: v.detach().cpu().float() for k, v in attn_scores.items()},
                    'genomic_features': genomic_features.detach().cpu().float(),
                    'pred_class':       pred_class,
                    'msi_prob':         msi_prob,
                    'true_label':       label.item()
                })
 
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true_list, y_pred_list)
    try:
        auc = roc_auc_score(y_true_list, y_score_list)
    except ValueError:
        auc = 0.5
 
    return avg_loss, acc, auc, batch_results
 
 
# =========================================================================
# [XAI 핵심 수정] A_coattn shape 자동 감지 및 안전 처리
# =========================================================================

 
def resolve_coattn_shape(A_coattn, n_omic=1425):
    """A_coattn을 무조건 (n_omic, N_patches) 2D로 변환."""
    
    # 1. 1425가 있는 차원을 찾아서 dim 0으로 이동 후 2D reshape
    for dim_idx in range(A_coattn.dim()):
        if A_coattn.shape[dim_idx] == n_omic:
            # 해당 dim을 맨 앞으로 이동
            A_coattn = A_coattn.transpose(0, dim_idx)
            # (1425, ...) 나머지 전부 합쳐서 2D
            A_coattn = A_coattn.reshape(n_omic, -1)
            return A_coattn
    
    # 1425를 못 찾은 경우 - 전체를 1행으로
    A_coattn = A_coattn.reshape(1, -1)
    return A_coattn


def build_xai_record(result, var_vocab_inv, vc_vocab_inv, func_vocab_inv, fold):
    patient_id       = result['patient_id']
    attn_scores      = result['attn_scores']
    genomic_features = result['genomic_features']
    pred_class       = result['pred_class']
    msi_prob         = result['msi_prob']
    true_lbl         = result['true_label']

    prediction_label = 'MSIMUT' if pred_class == 1 else 'MSS'
    prob_pct         = round(msi_prob * 100 if pred_class == 1 else (1 - msi_prob) * 100, 1)
    true_label_str   = 'MSIMUT' if true_lbl == 1 else 'MSS'

    # ✅ 핵심 수정: 1425 차원을 직접 찾아서 2D로 강제 변환
    A_coattn_2d = resolve_coattn_shape(attn_scores['coattn'], n_omic=1425)
    # 이제 shape 보장: (1425, N_patches)

    omic_importance  = A_coattn_2d.sum(dim=1)          # (1425,)
    top_omic_idx     = omic_importance.argmax().item()  # 0~1424

    # ✅ top_patch_idx도 안전하게 clamp
    patch_row        = A_coattn_2d[top_omic_idx]       # (N_patches,)
    top_patch_idx    = patch_row.argmax().item()        # 0~N_patches-1
    patch_attn_score = patch_row[top_patch_idx].item()  # ✅ 2D 인덱싱 대신 1D로

    # genomic_features 안전 인덱싱
    safe_omic_idx = min(top_omic_idx, genomic_features.shape[0] - 1)
    genomic_row   = genomic_features[safe_omic_idx]

    var_id   = int(genomic_row[0].item())
    vc_id    = int(genomic_row[1].item())
    func_ids = genomic_row[2:8].long()

    var_name = var_vocab_inv.get(var_id, f'Unknown_Var#{var_id}')
    vc_name  = vc_vocab_inv.get(vc_id,  f'Unknown_VC#{vc_id}')

    active_funcs = []
    for f_id in func_ids:
        f_id_val = int(f_id.item())
        if f_id_val > 0 and f_id_val in func_vocab_inv:
            active_funcs.append(func_vocab_inv[f_id_val])

    if active_funcs:
        func_str = ", ".join(active_funcs)
        reason = (f"{var_name}({vc_name}) 돌연변이의 [{func_str}] 기능이 "
                  f"WSI 패치#{top_patch_idx}의 형태 이상을 주목"
                  f"(Attention: {patch_attn_score:.4f}) -> {prediction_label} 판별에 기여")
    else:
        func_str = 'N/A'
        reason = (f"{var_name}({vc_name}) 돌연변이가 "
                  f"WSI 패치#{top_patch_idx}의 morphological 패턴 주목"
                  f"(Attention: {patch_attn_score:.4f}) -> {prediction_label} 판별에 기여")

    return {
        'fold':            fold,
        'patient':         patient_id,
        'true_label':      true_label_str,
        'prediction':      prediction_label,
        'correct':         'O' if true_label_str == prediction_label else 'X',
        'probability(%)':  prob_pct,
        'top_variant':     var_name,
        'variant_class':   vc_name,
        'gene_functions':  func_str,
        'top_patch_idx':   top_patch_idx,
        'attention_score': round(patch_attn_score, 4),
        'reason':          reason
    }
 
 
# =========================================================================
# 파이프라인 통제 사령부
# =========================================================================
def main():
    print("=== [MCAT 원 논문 재현] 5-Fold CV 학습 스크립트 시작 ===\n")
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> [시스템] 딥러닝 엔진: {device} | AMP 사용: {USE_AMP}")
 
    epoch_summary_log = []
    prediction_log    = []
    fold_test_log     = []
 
    # [경로 설정]
    clin_txt = '/home/team1/data/coad_msi_mss/common_patients.txt'
    mut_csv  = '/home/team1/cyl/VisionGen-COAD/cyl/genomicdata/preprocessed_mutation_data.csv'
    npy_file = '/home/team1/cyl/VisionGen-COAD/cyl/genomicdata/genomic_input_matrix.npy'
    pkl_file = '/home/team1/cyl/VisionGen-COAD/cyl/genomicdata/genomic_encoding_states.pkl'
    wsi_dir  = '/home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath'
 
    # 0. 단어 사전 로드
    print("-> 0/3 사전 데이터(.pkl) 확보 중...")
    try:
        with open(pkl_file, 'rb') as f:
            encoding_states = pickle.load(f)
        vocab_sizes = {
            'var':  len(encoding_states['var_vocab']),
            'vc':   len(encoding_states['vc_vocab']),
            'func': len(encoding_states['func_vocab'])
        }
        print(f"[성공] 사전 로딩 성공! (변이종류: {vocab_sizes['var']}개 등)")
 
        var_vocab_inv  = {v: k for k, v in encoding_states['var_vocab'].items()}
        vc_vocab_inv   = {v: k for k, v in encoding_states['vc_vocab'].items()}
        func_vocab_inv = {v: k for k, v in encoding_states['func_vocab'].items()}
 
    except FileNotFoundError:
        print("[경고] pkl 파일을 찾을 수 없습니다.")
        vocab_sizes    = {'var': 1500, 'vc': 25, 'func': 100}
        var_vocab_inv  = {}
        vc_vocab_inv   = {}
        func_vocab_inv = {}
 
    # 1. 데이터셋 로드
    print("-> 1/3 데이터로더 준비 중...")
    dataset = Pathomic_Classification_Dataset(
        clin_txt_path=clin_txt, mut_csv_path=mut_csv, npy_path=npy_file, data_dir=wsi_dir
    )
    # dataset = Subset(dataset, range(10))  # dry-run용
    print(f"[데이터] 전체 환자 수: {len(dataset)}명")
 
    # =========================================================================
    # [5-Fold CV 설정]
    #   KFold 5등분 → test=20% / trainval=80%
    #   trainval 중 20%를 val → train=64%, val=16%, test=20%
    # =========================================================================
    N_FOLDS    = 5
    MAX_EPOCHS = 100   # ✅ 수정: 충분한 epoch 확보 (ES_STOP_EPOCH 이후 Early Stop 작동)
    REG        = 1e-5
    loss_fn    = nn.CrossEntropyLoss()
 
    # ✅ [AMP] CUDA 있을 때만 GradScaler 생성
    scaler = GradScaler(device='cuda') if USE_AMP else None
 
    # ✅ [Early Stopping 설정 수정]
    ES_PATIENCE   = 10  # 10 epoch 연속 개선 없으면 중단
    ES_STOP_EPOCH = 5  # 최소 5 epoch 학습 후 Early Stop 허용 (너무 초반 노이즈로 멈추는 것 방지) (MAX_EPOCHS > 20이어야 의미있음)
    ES_WARMUP     = 0
 
    kf          = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    all_indices = list(range(len(dataset)))
 
    save_dir = './results_v5_4'
    os.makedirs(save_dir, exist_ok=True)
 
    # Fold Test 결과 실시간 저장 파일 준비
    fold_test_path = os.path.join(save_dir, 'fold_test_summary.csv')
    with open(fold_test_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'test_loss', 'test_acc', 'test_auc', 'stopped_epoch'])
        writer.writeheader()
 
    print(f"\n--- [MCAT 원 논문 방식] 5-Fold CV 시작 ---\n"
          f"    Optimizer    : Adam, lr=2e-4, weight_decay={REG}\n"
          f"    Data Split   : Train 64% / Val 16% / Test 20%\n"
          f"    EarlyStopping: patience={ES_PATIENCE}, stop_epoch={ES_STOP_EPOCH}\n"
          f"    Max Epochs   : {MAX_EPOCHS}\n"
          f"    AMP          : {USE_AMP}\n")
 
    for fold, (trainval_idx, test_idx) in enumerate(kf.split(all_indices)):
        print(f"\n{'='*55}")
        print(f" [Fold {fold+1}/{N_FOLDS}] 학습 시작")
        print(f"{'='*55}")
 
        model = MCAT_Single_Branch_Model(
            vocab_sizes=vocab_sizes, path_dim=1536, n_classes=2
        ).to(device)
 
        # ✅ 원 논문 Optimizer 그대로 (Adam + weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=REG)
 
        train_idx, val_idx = train_test_split(
            list(trainval_idx), test_size=0.2, random_state=42, shuffle=True
        )
 
        train_dataset = Subset(dataset, train_idx)
        val_dataset   = Subset(dataset, val_idx)
        test_dataset  = Subset(dataset, list(test_idx))
 
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)
 
        print(f"[데이터 분할] Train: {len(train_dataset)}명(64%) "
              f"| Val: {len(val_dataset)}명(16%) "
              f"| Test: {len(test_dataset)}명(20%)")
 
        early_stopping = EarlyStopping(
            warmup=ES_WARMUP, patience=ES_PATIENCE,
            stop_epoch=ES_STOP_EPOCH, verbose=True
        )
 
        stopped_epoch = MAX_EPOCHS  # Early Stop 없으면 max epoch
 
        # ============================
        # [Epoch 루프]
        # ============================
        for epoch in range(MAX_EPOCHS):
            train_loss, train_acc, train_auc, _ = run_epoch(
                model, train_loader, loss_fn, device,
                optimizer=optimizer, is_train=True, scaler=scaler
            )
            val_loss, val_acc, val_auc, _ = run_epoch(
                model, val_loader, loss_fn, device, is_train=False
            )
 
            epoch_summary_log.append({
                'fold':       fold + 1,
                'epoch':      epoch + 1,
                'train_loss': round(train_loss, 4),
                'train_acc':  round(train_acc * 100, 2),
                'train_auc':  round(train_auc, 4),
                'val_loss':   round(val_loss, 4),
                'val_acc':    round(val_acc * 100, 2),
                'val_auc':    round(val_auc, 4),
            })
 
            print(f"  [Fold {fold+1} | Epoch {epoch+1:03d}/{MAX_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% AUC: {val_auc:.4f}")
 
            early_stopping(epoch, val_loss, model)
            if early_stopping.early_stop:
                stopped_epoch = epoch + 1
                print(f"  [Early Stop] Fold {fold+1} → Epoch {stopped_epoch}에서 조기 종료")
                break
 
        # ✅ 최적 가중치 복원
        if early_stopping.best_state is not None:
            model.load_state_dict(early_stopping.best_state)
            print(f"  [복원] Best Val Loss={early_stopping.val_loss_min:.4f} 시점 가중치 복원 완료")
 
        # ============================
        # [최종 Test - Fold당 딱 1번]
        # ============================
        print(f"\n  -> [Fold {fold+1}] 최종 Test 평가 중...")
        test_loss, test_acc, test_auc, test_batch_results = run_epoch(
            model, test_loader, loss_fn, device, is_train=False, save_xai=True
        )
 
        print(f"\n{'='*55}")
        print(f" [Fold {fold+1} 최종 Test 결과]")
        print(f"  Test Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}% | AUC: {test_auc:.4f}")
        print(f"  (stopped at epoch {stopped_epoch})")
        print(f"{'='*55}\n")
 
        # XAI 기록
        for result in test_batch_results:
            record = build_xai_record(result, var_vocab_inv, vc_vocab_inv, func_vocab_inv, fold+1)
            record['test_loss']     = round(test_loss, 4)
            record['test_acc']      = round(test_acc * 100, 2)
            record['test_auc']      = round(test_auc, 4)
            record['stopped_epoch'] = stopped_epoch
            prediction_log.append(record)
 
        # Fold 결과 실시간 저장
        new_row = {
            'fold':          fold + 1,
            'test_loss':     round(test_loss, 4),
            'test_acc':      round(test_acc * 100, 2),
            'test_auc':      round(test_auc, 4),
            'stopped_epoch': stopped_epoch,
        }
        fold_test_log.append(new_row)
        with open(fold_test_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['fold', 'test_loss', 'test_acc', 'test_auc', 'stopped_epoch'])
            writer.writerow(new_row)
        print(f"  [CSV 저장] Fold {fold+1} Test 결과 기록 완료")
 
        # GPU 메모리 정리
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
 
    # =========================================================================
    # [최종 저장]
    # =========================================================================
    summary_path = os.path.join(save_dir, 'epoch_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_summary_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_summary_log)
    print(f"[저장 완료] Epoch 요약 -> {summary_path}")
 
    pred_path = os.path.join(save_dir, 'MSI_MSS_prediction.csv')
    with open(pred_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            'fold', 'patient', 'true_label', 'prediction', 'correct', 'probability(%)',
            'top_variant', 'variant_class', 'gene_functions',
            'top_patch_idx', 'attention_score',
            'test_loss', 'test_acc', 'test_auc', 'stopped_epoch', 'reason'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_log)
    print(f"[저장 완료] MSI/MSS 예측 (XAI) -> {pred_path}")
 
    # 최종 요약 출력
    if fold_test_log:
        all_test_auc = [r['test_auc'] for r in fold_test_log]
        all_test_acc = [r['test_acc'] for r in fold_test_log]
        print(f"\n{'='*55}")
        print(f" [5-Fold 최종 요약]")
        print(f"  Test AUC 평균: {sum(all_test_auc)/len(all_test_auc):.4f}")
        print(f"  Test Acc 평균: {sum(all_test_acc)/len(all_test_acc):.2f}%")
        for r in fold_test_log:
            print(f"  Fold {r['fold']}: AUC={r['test_auc']:.4f} | Acc={r['test_acc']:.2f}% | "
                  f"Stopped Epoch={r['stopped_epoch']}")
        print(f"{'='*55}\n")
 
    print("[성공] REDCAB_MCAT 5-Fold CV 학습 및 분석 완료!!!")
 
 
if __name__ == '__main__':
    main()
