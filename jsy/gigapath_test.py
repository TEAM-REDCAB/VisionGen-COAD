import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve,
    confusion_matrix
)

from gigapath.classification_head import ClassificationHead
from h5dataset import H5Dataset
import config as cf
# --- 설정 및 경로 (사용자 환경에 맞게 수정) ---
RESULTS_PATH = cf.get_results_path()
MODEL_PATH = os.path.join(RESULTS_PATH, 'saved_models')
TEST_RESULT_DIR = os.path.join(cf.get_results_path(), 'test_results')
os.makedirs(TEST_RESULT_DIR, exist_ok=True)

# 시드 고정
np.random.seed(cf.SEED)
torch.manual_seed(cf.SEED)

def test_gigapath_full_slide():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_fold_results = []
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    total_cm = np.zeros((2, 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for fold in tqdm(range(5), desc="Fold"):
        print(f"\n🚀 Testing GigaPath Fold {fold} (Full Slide Inference)...")
        current_fold_col = f'fold_{fold}'
        
        # 테스트셋 로드 (전체 패치 사용)
        test_dataset = H5Dataset(split="test", fold_col=current_fold_col)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 1. 모델 빌드 및 베스트 가중치 로드
        model = ClassificationHead(input_dim=1536, latent_dim=768, feat_layer="5-11", n_classes=1).to(device)
        model_path = os.path.join(MODEL_PATH, f'vit_gigapath_fold_{fold}_best.pth')
        
        if not os.path.exists(model_path):
            print(f"⚠️ {model_path}를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        all_labels, all_probs = [], []
        
        with torch.no_grad():
            for features, coords, labels in tqdm(test_loader, desc='loader'):
                # [1, L, 1536], [1, L, 2] 형태로 바로 GPU 전송
                features = features.to(device)
                coords = coords.to(device)
                
                # --- 전체 슬라이드 단일 추론 ---
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    # 청킹 없이 전체 데이터를 한 번에 모델에 입력
                    logits = model(features, coords).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        # 지표 계산
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        preds = (all_probs >= 0.5).astype(int)
        acc = np.mean(preds == all_labels)
        
        cm = confusion_matrix(all_labels, preds, labels=[0, 1])
        total_cm += cm
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        all_fold_results.append({
            'Fold': fold, 'AUC': auc, 'AUPRC': auprc, 
            'Accuracy':acc, 'Sensitivity': sensitivity, 'Specificity': specificity
        })

        # 시각화용 데이터 누적
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)

        ax1.plot(fpr, tpr, color=colors[fold], lw=1.5, alpha=0.3, label=f'Fold {fold} ({auc:.3f})')
        
        # PR Curve 그리기
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        ax2.plot(recall, precision, color=colors[fold], lw=1.5, alpha=0.3)
        
        print(f"✅ Fold {fold} 완료: AUC={auc:.4f}, AUPRC={auprc:.4f}")
        # [추가] 현재 폴드 모델과 옵티마이저 명시적 삭제
        del model
        
        # [추가] 가비지 컬렉션 및 GPU 캐시 강제 비우기
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"🧹 Fold {fold} 메모리 정리 완료.")

    # --- 최종 결과 도출 및 시각화 ---
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # ROC Curve 마무리
    ax1.plot(mean_fpr, mean_tpr, color='black', lw=3, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax1.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tprs, 0), 0), np.minimum(mean_tpr + np.std(tprs, 0), 1), color='grey', alpha=0.2)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('GigaPath Mean ROC Curves')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # PR Curve 마무리
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('GigaPath Precision-Recall Curves')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(TEST_RESULT_DIR, 'gigapath_combined_curves.png'), dpi=300)

    # 4. 합산 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_cm, annot=True, fmt='g', cmap='Oranges', 
                xticklabels=['MSS (0)', 'MSI (1)'], yticklabels=['MSS (0)', 'MSI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('GigaPath Aggregated Confusion Matrix (Full Slide)')
    plt.savefig(os.path.join(TEST_RESULT_DIR, 'gigapath_total_confusion_matrix.png'), dpi=300)

    # 최종 요약 리포트 저장
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(os.path.join(TEST_RESULT_DIR, 'gigapath_test_summary.csv'), index=False)
    
    print(f"\n{'='*20} GigaPath Test Summary {'='*20}")
    print(results_df.to_string(index=False))
    print(f"\n⭐ Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"💾 모든 결과가 {TEST_RESULT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    test_gigapath_full_slide()