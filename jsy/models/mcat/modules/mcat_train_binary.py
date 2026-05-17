import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm
import numpy as np


def train_binary(epoch, model, loader, optimizer, loss_fn, gc=16, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = 0.0
    all_labels = []
    all_probs = []

    optimizer.zero_grad()

    pbar = tqdm(
        loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, dynamic_ncols=True
    )

    for batch_idx, (data_wsi, data_omic, label) in enumerate(pbar):
        data_wsi = data_wsi.to(device)
        data_omic = data_omic.to(device)
        label = label.type(torch.FloatTensor).to(device)

        logits, *_ = model(data_wsi, data_omic)

        logits = logits.squeeze(dim=-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)

        loss = loss_fn(logits, label)
        loss_value = loss.item()

        loss = loss / gc
        loss.backward()

        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss_value

        probs = torch.sigmoid(logits)
        all_labels.extend(label.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({"Loss": f"{loss_value:.4f}"})

    avg_loss = train_loss / len(loader)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    # 에폭 단위 결과 출력
    print(
        f"====> Epoch: {epoch:02d} | Train Loss: {avg_loss:.4f} | Train AUROC: {auroc:.4f} | Train AUPRC: {auprc:.4f} <===="
    )

    return all_labels, all_probs, auroc, auprc


def validate_binary(epoch, model, loader, loss_fn, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_loss = 0.0
    all_labels = []
    all_probs = []

    pbar = tqdm(
        loader, desc=f"Epoch {epoch:02d} [Valid]", leave=False, dynamic_ncols=True
    )

    with torch.no_grad():
        for data_wsi, data_omic, label in pbar:
            data_wsi = data_wsi.to(device)
            data_omic = data_omic.to(device)
            label = label.type(torch.FloatTensor).to(device)

            logits, *_ = model(data_wsi, data_omic)

            logits = logits.squeeze(dim=-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = val_loss / len(loader)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    # 에폭 단위 결과 출력
    print(
        f"====> Epoch: {epoch:02d} | Valid Loss: {avg_loss:.4f} | Valid AUROC: {auroc:.4f} | Valid AUPRC: {auprc:.4f} <===="
    )

    # return avg_loss, auroc, auprc, f1, best_thresh
    return all_labels, all_probs, auroc, auprc
