import torch
import os
import numpy as np
import glob
import h5py
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report, average_precision_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 기존에 작성한 모듈 임포트
from modules.mcat_multimodal_dataset import MSI_Multimodal_Dataset, get_label_path
from modules.mcat_model import MCAT_Binary


def save_attention_heatmap(patient_id, attn_scores, label, prob, coords_array, save_dir, mode='absolute'):
    """
    배열을 직접 입력받아 그리는 가장 직관적이고 오류 없는 시각화 함수
    """
    # 길이 맞춤 (이제 무조건 일치하지만 안전장치 유지)
    valid_len = min(len(coords_array), len(attn_scores))
    coords_array = coords_array[:valid_len]
    attn_scores = attn_scores[:valid_len]

    # =====================================================================
    # 2. 시각화 (ABMIL scatter 방식)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')

    # 💡 1. 모드에 따른 스케일링 결정
    if mode == 'relative':
        # 해당 슬라이드 내부의 99% 값을 기준으로 (내적 분포 강조)
        plot_scores = attn_scores
        vmin = np.min(plot_scores)
        vmax = np.percentile(plot_scores, 99)
        cb_label = 'Relative Attention Score'
    else:
        # 제곱근 스케일링 및 전 슬라이드 공통 기준 적용 (외적 강도 비교)
        mean_val = np.mean(attn_scores)
        norm_scores = attn_scores / (mean_val + 1e-8)
        plot_scores = np.sqrt(norm_scores)
        vmin, vmax = 0.5, 2.1 # 사용자님이 찾으신 최적의 그라데이션 범위
        cb_label = 'Standardized Importance (Square Root Scale)'

    # 💡 2. 렌더링
    scatter = ax.scatter(
        coords_array[:, 0], -coords_array[:, 1], 
        c=plot_scores, cmap='jet', s=25, 
        vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none', marker='s'
    )
    
    plt.colorbar(scatter, ax=ax, label=cb_label)

    # =====================================================================
    # 3. 텍스트 및 제목 설정
    # =====================================================================
    label_str = "MSI" if label == 1 else "MSS"
    is_correct = (prob >= 0.5 and label == 1) or (prob < 0.5 and label == 0)
    
    title_obj = plt.title(f"Patient ID: {patient_id} | True: {label_str} | Prob(MSI): {prob:.4f}", fontsize=18, pad=20)
    
    if not is_correct:
        plt.setp(title_obj, color='red')
    
    plt.axis('equal') 
    plt.axis('off')   

    save_path = os.path.join(save_dir, f"P{prob:.2f}_{label_str}_{patient_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  📸 생성 완료: {os.path.basename(save_path)}")


def evaluate_test_set_with_viz(result_path, csv_path, feats_path, npy_path, pkl_path, coords_dir, thumbs_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====> 실전 테스트 시작 <====")
    
    test_dataset = MSI_Multimodal_Dataset(
        split='test', 
        fold_col='fold_0', 
        csv_path=csv_path,
        feats_path=feats_path,
        npy_path=npy_path,
        pkl_path=pkl_path
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"테스트 환자 수: {len(test_dataset)}명\n")

    model = MCAT_Binary().to(device)
    model.eval() 
    
    all_labels =[]
    ensemble_probs = None
    sum_thresh = 0.0

    viz_save_dir = "./visualization/heatmaps_png"
    os.makedirs(viz_save_dir, exist_ok=True)
    
    ensemble_attn_dict = {}

    for fold in range(1, 6):
        model_path = os.path.join(result_path, f"best_model_fold{fold}.pt")
        print(f"로드할 모델 가중치: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        best_thresh = checkpoint['best_thresh']
        sum_thresh += best_thresh
        print(f"  - 이 모델의 Best Threshold: {best_thresh:.4f}")

        fold_labels = []
        fold_probs = []

        pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
        
        # save_attention_heatmap.py 의 추론 루프 안

        with torch.no_grad(): 
            for i, (data_wsi, data_omic, label) in enumerate(pbar):
                data_wsi = data_wsi.to(device)
                data_omic = data_omic.to(device)
                label = label.type(torch.FloatTensor).to(device)
                
                # 💡 수정: A_path도 함께 받습니다.
                logits, A_coattn, A_path = model(data_wsi, data_omic)

                logits = logits.squeeze(dim=-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                probs = torch.sigmoid(logits)
                fold_probs.extend(probs.cpu().numpy())

                patient_id = test_dataset.df.iloc[i]['patient']
                
                # 🔥 완벽한 어텐션 압축: 행렬 곱셈 (Weighted Sum) 🔥
                # A_path   : (1, 1, 1425) -> 1425개 유전체 가이드 피처의 중요도
                # A_coattn : (1, 1425, N) -> 각 유전체 피처가 N개 패치에 준 가중치
                # torch.bmm 결과 -> (1, 1, N) 
                final_attention = torch.bmm(A_path, A_coattn)
                
                # Squeeze를 통해 (N,) 1차원 배열로 만듭니다.
                attn_patch_scores = final_attention.squeeze().cpu().numpy()

                # ----------------------------------------------------
                # 🚨 [디버깅용 출력] 첫 번째 배치의 데이터만 확인해보기
                if i == 0 and fold == 1:
                    print("\n[디버깅 로그]")
                    
                    # 1. A_path가 한쪽에 쏠려 있는지(Spike), 평평한지(Uniform) 확인
                    a_path_arr = A_path.squeeze().cpu().numpy()
                    print(f"👉 A_path (유전체 가중치) -> Max: {a_path_arr.max():.5f}, Min: {a_path_arr.min():.5f}, Mean: {a_path_arr.mean():.5f}")
                    
                    # 2. bmm과 np.mean 결과가 수치상으로도 아예 똑같은지 확인
                    mean_scores = np.mean(A_coattn.squeeze().cpu().numpy(), axis=0)
                    diff = np.abs(attn_patch_scores - mean_scores).max()
                    print(f"👉 np.mean과 torch.bmm 점수의 최대 차이: {diff:.8f}")
                    print("----------------------------------------------------\n")
                # ----------------------------------------------------

                # (이하 기존 코드와 동일)
                if fold == 1:
                    fold_labels.extend(label.cpu().numpy())
                    ensemble_attn_dict[patient_id] = {
                        'attention': attn_patch_scores, # 완벽하게 계산된 (N,)
                        'label': label.item()
                    }
                else:
                    ensemble_attn_dict[patient_id]['attention'] += attn_patch_scores

        fold_probs = np.array(fold_probs)

        if ensemble_probs is None:
            ensemble_probs = fold_probs
            all_labels = np.array(fold_labels)
        else:
            ensemble_probs += fold_probs
        
    ensemble_probs = ensemble_probs / 5
    ensemble_thresh = sum_thresh / 5
    print(f"\n👉 앙상블에 적용될 평균 Threshold: {ensemble_thresh:.4f}")

    # ==============================================================================
    # 🔥 [자동 시각화 루프] 앙상블 계산이 끝난 후 전 환자 히트맵 생성
    # ==============================================================================
    print("\n\n📸 전체 환자의 앙상블 어텐션 히트맵 이미지를 생성하고 저장합니다...")
    
    vis_pbar = tqdm(enumerate(ensemble_attn_dict.items()), total=len(ensemble_attn_dict), desc="Visualizing", dynamic_ncols=True)
    
    for idx, (patient_id, data) in vis_pbar:
        avg_attn = data['attention'] / 5.0
        patient_prob = float(ensemble_probs[idx]) 
        
        # 🔥 핵심 수정: Dataset이 피처를 불러올 때 썼던 파일 목록을 똑같이 가져옵니다.
        matching_files = test_dataset.patient_to_files.get(patient_id, [])
        if not matching_files:
            print(f"\n⚠️ [건너뜀] {patient_id} 환자의 특징 파일을 찾을 수 없습니다.")
            continue
            
        # 피처와 완벽하게 1:1 대응되는 좌표(coords) 배열 조립
        all_coords = []
        for feat_fp in matching_files:
            try:
                with h5py.File(feat_fp, "r") as f:
                    all_coords.append(f["coords"][:])
            except Exception as e:
                print(f"❌ 좌표 로드 에러 ({feat_fp}): {e}")
                continue
                
        if not all_coords:
            continue
            
        coords_array = np.concatenate(all_coords, axis=0)
        
        # 도우미 함수 호출 (파일 경로 대신 좌표 배열을 직접 전달)
        save_attention_heatmap(
            patient_id=patient_id,
            attn_scores=avg_attn,
            label=data['label'],
            prob=patient_prob,          
            coords_array=coords_array,  # <--- 직접 조립한 완벽한 좌표 배열!
            save_dir=viz_save_dir
        )
            
    print(f"✅ 총 {len(ensemble_attn_dict)}명 환자의 히트맵 이미지 저장 완료! (경로: {viz_save_dir})\n")

    # 5. 임계값(Threshold) 적용 및 최종 성능 지표 계산
    auroc = roc_auc_score(all_labels, ensemble_probs)
    auprc = average_precision_score(all_labels, ensemble_probs)
    
    ensemble_preds = (ensemble_probs >= ensemble_thresh).astype(float)
    
    f1 = f1_score(all_labels, ensemble_preds, zero_division=0)
    cm = confusion_matrix(all_labels, ensemble_preds)

    report = classification_report(all_labels, ensemble_preds, target_names=["MSS (0)", "MSI-H (1)"], zero_division=0)

    print("\n" + "="*40)
    print("🏆 5-Fold 앙상블 최종 테스트 결과 🏆")
    print("="*40)
    print(f"Ensemble AUROC : {auroc:.4f}")
    print(f"Ensemble AUPRC : {auprc:.4f}")
    print(f"Ensemble F1    : {f1:.4f}\n")
    
    print("[Confusion Matrix]")
    print(" TN(MSS정답)  FP(MSI오진)")
    print(" FN(MSS오진)  TP(MSI정답)")
    print(cm)
    
    print("\n[Classification Report]")
    print(report)

if __name__ == '__main__':
    PATIENTS_LABEL = "./data/common_patients.txt"
    CSV_PATH = get_label_path(PATIENTS_LABEL) 
    FEATS_PATH = "/home/team/projects/team_REDCAB/team_project/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath"
    NPY_PATH = "./data/genomic_input_matrix.npy"
    PKL_PATH = "./data/genomic_encoding_states.pkl"
    RESULT_PATH = "./results_msi"
    
    COORDS_DIR = "/home/team/projects/team_REDCAB/team_project/data/gigapath_processed/20.0x_256px_0px_overlap/patches"
    THUMBS_DIR = "/home/team/projects/team_REDCAB/team_project/data/gigapath_processed/thumbnails"
    
    evaluate_test_set_with_viz(
        RESULT_PATH, CSV_PATH, FEATS_PATH, NPY_PATH, PKL_PATH, COORDS_DIR, THUMBS_DIR
    )