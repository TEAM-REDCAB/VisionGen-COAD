import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from modules.mcat_multimodal_dataset import (
    MSI_Multimodal_Dataset,
    get_label_path,
    get_feats_path,
)
from modules.mcat_model import MCAT_Binary
from modules.mcat_train_binary import train_binary, validate_binary
import numpy as np

SEED = 42


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1. 하이퍼파라미터 설정
    max_epochs = 20
    batch_size = 1
    lr = 5e-5
    gc_steps = 16  # VRAM 12GB 맞춤형 Gradient Accumulation
    results_dir = "./results_msi"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 전체 데이터셋 로드 (이전 단계에서 만든 MSI_Multimodal_Dataset)
    common_patients = "./data/common_patients.txt"  # 경로 수정 필요
    feats_path = get_feats_path()
    npy_path = "./data/genomic_input_matrix.npy"
    pkl_path = "./data/genomic_encoding_states.pkl"

    labeled_csv_path = get_label_path(common_patients)
    oof_probs = None
    oof_labels = None

    # 4. 5-Fold 교차 검증 루프 (새로운 데이터셋 구조 적용)
    for fold_idx in range(5):
        fold_col = f"fold_{fold_idx}"
        print(f"\n{'=' * 40}")
        print(f"========== Fold {fold_idx} 학습 시작 ==========")
        print(f"{'=' * 40}")

        # Train 및 Val 데이터셋 선언 (이미 나누어진 fold_col 기준)
        train_dataset = MSI_Multimodal_Dataset(
            split="train",
            fold_col=fold_col,
            csv_path=labeled_csv_path,  # 새로 생성된 CSV 사용
            feats_path=feats_path,
            npy_path=npy_path,
            pkl_path=pkl_path,
        )

        val_dataset = MSI_Multimodal_Dataset(
            split="val",
            fold_col=fold_col,
            csv_path=labeled_csv_path,
            feats_path=feats_path,
            npy_path=npy_path,
            pkl_path=pkl_path,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=lambda _: np.random.seed(SEED),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=lambda _: np.random.seed(SEED),
        )

        # 모델 및 옵티마이저 초기화 (폴드마다 가중치 초기화)
        model = MCAT_Binary().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )
        n_neg = int((train_dataset.df["msi"] == 0).sum())
        n_pos = int((train_dataset.df["msi"] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(
            device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auprc = 0.0
        best_val_auroc = 0.0
        best_val_probs = None
        best_val_labels = None

        # --- 🚨 얼리 스토핑 관련 변수 추가 ---
        patience = 5  # 최고 성능을 갱신하지 못하고 5에폭을 버티면 종료
        early_stop_counter = 0

        for epoch in range(1, max_epochs + 1):
            train_binary(epoch, model, train_loader, optimizer, loss_fn, gc=gc_steps)
            val_labels, val_probs, val_auroc, val_auprc = validate_binary(
                epoch, model, val_loader, loss_fn
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
                    results_dir, f"best_model_fold{fold_idx}.pt"
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

    # 최종 결과 출력
    print("\n================ 최종 결과 ================")
    fpr, tpr, roc_thresholds = roc_curve(oof_labels, oof_probs)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    oof_best_thresh = roc_thresholds[best_idx]

    print(f"\nOOF 최적 Threshold (Youden's Index): {oof_best_thresh:.4f}")

    # 💡 5. 앙상블 최종 지표 계산
    oof_auroc = roc_auc_score(oof_labels, oof_probs)
    oof_auprc = average_precision_score(oof_labels, oof_probs)
    oof_preds = (oof_probs >= oof_best_thresh).astype(int)
    acc = np.mean(oof_preds == oof_labels)
    f1 = f1_score(oof_labels, oof_preds, zero_division=0)

    cm = confusion_matrix(oof_labels, oof_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # --- 결과 출력 ---
    print("\n" + "=" * 40)
    print("========== 🏆 OOF 최종 테스트 결과 ==========")
    print("=" * 40)
    print(f"OOF AUROC          : {oof_auroc:.4f}")
    print(f"OOF AUPRC          : {oof_auprc:.4f}")
    print(f"OOF F1-Score       : {f1:.4f}")
    print(f"OOF Threshold      : {oof_best_thresh:.4f} ")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Sensitivity(Recall): {sensitivity:.4f}")
    print(f"Specificity        : {specificity:.4f}")

    print("\n[Confusion Matrix]")
    print(" TN(MSS를맞춤)  FP(MSI로오해)")
    print(" FN(MSS로오해)  TP(MSI를맞춤)")
    print(cm)

    report = classification_report(
        oof_labels,
        oof_preds,
        target_names=["MSS (0)", "MSI-H (1)"],
        zero_division=0,
    )
    print("\n[Classification Report]")
    print(report)

    # 💡 6. 시각화 (앙상블된 '단 하나의 완벽한 선'만 그립니다)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ROC Curve 그리기
    ax1.plot(fpr, tpr, color="red", lw=2, label=f"OOF ROC (AUC = {oof_auroc:.4f})")
    ax1.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--", alpha=0.5)
    # 최적의 Threshold 위치에 별표(★) 표시
    ax1.scatter(
        fpr[best_idx],
        tpr[best_idx],
        marker="*",
        color="blue",
        s=200,
        label=f"Best Thresh ({oof_best_thresh:.4f})",
        zorder=5,
    )
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("OOF ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Precision-Recall Curve 그리기
    precision, recall, _ = precision_recall_curve(oof_labels, oof_probs)
    ax2.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"OOF PR (AUC = {oof_auprc:.4f})",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("OOF Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "curves_MCAT_oof.png"), dpi=300)
    plt.show()

    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["MSS (0)", "MSI (1)"],
        yticklabels=["MSS (0)", "MSI (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("OOF Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix_MCAT_oof.png"), dpi=300)
    plt.show()

    # 히스토그램 시각화 (앙상블 확률 기준)
    plt.figure(figsize=(10, 6))
    mss_probs = oof_probs[oof_labels == 0]
    msi_probs = oof_probs[oof_labels == 1]

    sns.histplot(mss_probs, bins=50, color="blue", alpha=0.6, label="MSS (0)", kde=True)
    sns.histplot(
        msi_probs, bins=50, color="red", alpha=0.6, label="MSI-H (1)", kde=True
    )

    # 히스토그램 위에 새로운 Threshold를 검은 점선으로 표시하여 직관성 확보
    plt.axvline(
        oof_best_thresh,
        color="black",
        linestyle="--",
        lw=2,
        label=f"Threshold ({oof_best_thresh:.4f})",
    )

    plt.xlabel("OOF Predicted Probability")
    plt.ylabel("Count of Patients")
    plt.title("Distribution of OOF Probabilities")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "histogram_MCAT_oof.png"), dpi=300)
    plt.show()

    # 💡 7. 최종 결과 CSV 한 줄로 요약 저장
    results_dict = {
        "AUROC": [oof_auroc],
        "AUPRC": [oof_auprc],
        "F1_Score": [f1],
        "Accuracy": [acc],
        "Sensitivity": [sensitivity],
        "Specificity": [specificity],
        "Threshold": [oof_best_thresh],
    }
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(
        os.path.join(results_dir, "test_results_MCAT_oof.csv"), index=False
    )
    print(f"\n💾 OOF 최종 결과가 {results_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
