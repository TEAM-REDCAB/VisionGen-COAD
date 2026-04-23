"""
train.py
run_epoch() : train / eval 공통 루프 (cyl의 main_val_v5_4.py 기반)

변경점:
    - model 인자가 MCATTeacher로 교체되어도 동일하게 동작
    - XAI 기록은 main_teacher.py에서 처리 (관심사 분리)
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from tqdm import tqdm

def run_epoch(model, loader, loss_fn, device, optimizer=None, is_train=True, scaler=None, save_xai=False):
    """
    한 에폭 실행.
    반환: avg_loss, acc, auc, batch_results

    batch_results 원소:
        patient_id, attn_scores, genomic_features,
        pred_class, msi_prob, true_label
    """
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    y_true_list   = []
    y_score_list  = []
    y_pred_list   = []
    batch_results = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="Processing batches"):
            path_features, genomic_features, label, patient_id = batch

            # batch_size=1 기준: squeeze(0)으로 배치 차원 제거
            path_features    = path_features.squeeze(0).to(device)
            genomic_features = genomic_features.squeeze(0).to(device)
            label            = label.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                logits, Y_hat, attn_scores = model(path_features, genomic_features)
                loss = loss_fn(logits, label)

            total_loss += loss.item()

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # 수치 연산은 FP32로
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
                    'true_label':       label.item(),
                })

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true_list, y_pred_list)
    try:
        auc = roc_auc_score(y_true_list, y_score_list)
    except ValueError:
        auc = 0.5   # 단일 클래스만 있을 때 폴백

    return avg_loss, acc, auc, batch_results

    
