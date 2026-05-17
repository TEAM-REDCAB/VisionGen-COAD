import os
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
import numpy as np


def oof_results(oof_probs, oof_labels, results_dir, model_name):
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
    plt.savefig(os.path.join(results_dir, f"curves_{model_name}_oof.png"), dpi=300)
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
    plt.savefig(
        os.path.join(results_dir, f"confusion_matrix_{model_name}_oof.png"), dpi=300
    )
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
    plt.savefig(os.path.join(results_dir, f"histogram_{model_name}_oof.png"), dpi=300)
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
        os.path.join(results_dir, f"test_results_{model_name}_oof.csv"), index=False
    )
    print(f"\n💾 OOF 최종 결과가 {results_dir}에 저장되었습니다.")
