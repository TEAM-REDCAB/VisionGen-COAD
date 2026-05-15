import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def kd_loss_fn(
    s_logits,
    s_path_bag,
    s_attn,
    t_logits,
    t_path_bag,
    t_attn,
    labels,
    task_criterion,
    epoch,
    alpha,
    beta,
    gamma,
    w_task,
):
    """
    지식 증류 복합 손실 함수

    Args:
        s_logits, t_logits : (B,)  - sigmoid 이전 로짓
        s_path_bag         : (B, 256)
        t_path_bag         : (B, 256)
        s_attn, t_attn     : (B, 1, N) - 이미 softmax 완료된 어텐션
        labels             : (B,)
        alpha              : logit distillation 가중치
        beta               : attention distillation 가중치
        gamma              : path_bag distillation 가중치
        w_task             : task loss 가중치 (정답 anchor)
    """

    # ── 1. Task Loss ────────────────────────────────────────────────────────
    task_loss = task_criterion(s_logits, labels)

    # ── 2. Logit Distillation : BCE ─────────────────────────────────────────
    # 이진 분류에서 temperature T는 dark knowledge 효과가 없고
    # T² 배율이 logit loss 스케일을 폭탄으로 만드므로 완전 제거.
    # "학생의 MSI 확률이 티처의 MSI 확률을 따라가게 해라"로 단순화.
    s_prob = torch.sigmoid(s_logits)
    t_prob = torch.sigmoid(t_logits)
    if t_prob.shape != s_prob.shape:
        t_prob = t_prob.view_as(s_prob)
    logit_loss = F.binary_cross_entropy(s_prob, t_prob.detach())

    # ── 3. Attention Distillation : KL Divergence ───────────────────────────
    # MSE는 패치 수(N)가 가변(969~34618)이면 스케일이 1/N²으로 소멸함.
    # KL-Div는 확률분포 간 거리를 측정하므로 N과 무관하게 안정적인 스케일 유지.
    if t_attn.shape != s_attn.shape:
        t_attn = t_attn.view_as(s_attn)
    s_attn_flat = s_attn.squeeze(1)  # (B, N)
    t_attn_flat = t_attn.squeeze(1).detach()  # (B, N)
    attn_loss = F.kl_div(
        torch.log(s_attn_flat + 1e-8), t_attn_flat, reduction="batchmean"
    )

    # ── 4. Path Bag Distillation : 차원 정규화된 MSE ────────────────────────
    # 256차원 벡터의 MSE 합산값은 스케일이 크므로 차원 수(d_model)로 나눔.
    if t_path_bag.shape != s_path_bag.shape:
        t_path_bag = t_path_bag.view_as(s_path_bag)
    path_loss = F.mse_loss(s_path_bag, t_path_bag.detach())

    total_loss = (
        (w_task * task_loss)
        + (alpha * logit_loss)
        + (beta * attn_loss)
        + (gamma * path_loss)
    )
    return total_loss, task_loss, logit_loss, attn_loss, path_loss


def train_binary(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    gc=16,
    alpha=0.05,
    beta=0.5,
    gamma=0.05,
    w_task=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = 0.0
    all_labels = []
    all_probs = []

    # 에폭 단위 손실 누적 (에폭 요약 출력용)
    sum_task = sum_logit = sum_attn = sum_path = 0.0
    latent_grad_norms = []

    optimizer.zero_grad()

    pbar = tqdm(
        loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, dynamic_ncols=True
    )

    for batch_idx, (
        features,
        coords,
        labels,
        t_logits,
        t_path_bag,
        t_attn,
    ) in enumerate(pbar):
        features = features.to(device)
        labels = labels.type(torch.FloatTensor).to(device)
        t_logits = t_logits.to(device)
        t_path_bag = t_path_bag.to(device)
        t_attn = t_attn.to(device)

        s_logits, s_path_bag, s_attn = model(features)

        # (B, 1) → (B,) 로 squeeze
        s_logits = s_logits.squeeze(-1)
        if s_logits.dim() == 0:
            s_logits = s_logits.unsqueeze(0)

        total_loss, task_loss, logit_loss, attn_loss, path_loss = kd_loss_fn(
            s_logits,
            s_path_bag,
            s_attn,
            t_logits,
            t_path_bag,
            t_attn,
            labels,
            loss_fn,
            epoch,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_task=w_task,
        )

        loss_value = total_loss.item()
        (total_loss / gc).backward()

        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            if (
                hasattr(model, "latent_queries")
                and model.latent_queries.grad is not None
            ):
                latent_grad_norms.append(model.latent_queries.grad.norm().item())
            optimizer.zero_grad()

        train_loss += loss_value
        sum_task += task_loss.item()
        sum_logit += logit_loss.item()
        sum_attn += attn_loss.item()
        sum_path += path_loss.item()

        probs = torch.sigmoid(s_logits)
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        pbar.set_postfix(
            {
                "Loss": f"{loss_value:.3f}",
                "T": f"{task_loss.item():.3f}",
                "KD": f"{logit_loss.item():.3f}",
                "Att": f"{attn_loss.item():.4f}",
                "Bag": f"{path_loss.item():.4f}",
            }
        )

    n = len(loader)
    avg_loss = train_loss / n
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    # ────────────────────────────────────────────────────────────────────────
    auroc, auprc, best_thresh, f1 = predict_binary(all_probs, all_labels)

    # ── 에폭 단위 진단 요약 ─────────────────────────────────────────────────
    # 각 손실 항목의 평균을 출력해 어느 항목이 지배적인지 확인
    avg_latent_grad = np.mean(latent_grad_norms) if latent_grad_norms else 0.0
    pred_pos = (all_probs >= best_thresh).sum()
    pred_neg = n - pred_pos
    true_pos = int(all_labels.sum())
    true_neg = n - true_pos

    print(
        f"[Epoch {epoch:02d} 손실 비율] "
        f"Task={w_task} | Logit(alpha)={alpha} | Attn(beta)={beta} | Path(gamma)={gamma}"
    )
    print(
        f"[Epoch {epoch:02d} 손실 분해] "
        f"Task={sum_task / n:.4f} | Logit={sum_logit / n:.4f} | "
        f"Attn={sum_attn / n:.4f} | Path={sum_path / n:.4f}"
    )
    print(
        f"[Epoch {epoch:02d} 진단]      "
        f"latent_queries grad={avg_latent_grad:.6f} | "
        f"예측 MSI/MSS={pred_pos}/{pred_neg} | "
        f"실제 MSI/MSS={true_pos}/{true_neg}"
    )
    print(
        f"====> Epoch: {epoch:02d} | Train Loss: {avg_loss:.4f} "
        f"| AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} "
        f"| F1: {f1:.4f} | Thresh: {best_thresh:.4f} <===="
    )
    return auroc, auprc, best_thresh


def validate_binary(epoch, model, loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_loss = 0.0
    all_labels = []
    all_probs = []

    pbar = tqdm(
        loader, desc=f"Epoch {epoch:02d} [Valid]", leave=False, dynamic_ncols=True
    )

    with torch.no_grad():
        for features, coords, labels in pbar:
            features = features.to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            logits, *_ = model(features)
            logits = logits.squeeze(-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = loss_fn(logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = val_loss / len(loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auroc, auprc, best_thresh, f1 = predict_binary(all_probs, all_labels)
    print(
        f"====> Epoch: {epoch:02d} | Valid Loss: {avg_loss:.4f} "
        f"| AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} "
        f"| F1: {f1:.4f} | Thresh: {best_thresh:.4f} <===="
    )
    return auroc, auprc, best_thresh


def predict_binary(all_probs, all_labels, threshold_method="youden"):
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)

        if threshold_method == "youden":
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            index = tpr - fpr
        else:
            precisions, recalls, thresholds = precision_recall_curve(
                all_labels, all_probs
            )
            index = (
                2
                * (precisions[:-1] * recalls[:-1])
                / (precisions[:-1] + recalls[:-1] + 1e-8)
            )

        best_idx = np.argmax(index)
        best_thresh = thresholds[best_idx]

    except ValueError:
        auroc = 0.5
        auprc = 0.0
        best_thresh = 0.5

    all_preds = (all_probs >= best_thresh).astype(float)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return auroc, auprc, best_thresh, f1
