import os
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transmil_model import BinaryClassificationModel
import transmil_model as tm

# 1. 설정 및 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_PATH = tm.get_label_path()
MODEL_PATH = os.path.join(tm.get_results_path(), 'saved_models')
FEATS_PATH = tm.get_features_path()
OUTPUT_DIR = os.path.join(tm.get_results_path(), 'visualizations_full_slide')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PATCHES = tm.MAX_PATCHES # 4096

df = pd.read_csv(LABEL_PATH)

# 파일 목록 스캔 
print("📂 파일 목록 스캔 중...")
all_h5_files = [f for f in os.listdir(FEATS_PATH) if f.endswith('.h5')]
file_map = {}
for f in all_h5_files:
    prefix = f[:12]
    if prefix not in file_map:
        file_map[prefix] = []
    file_map[prefix].append(os.path.join(FEATS_PATH, f))

# 2. 시각화 실행
for fold in range(5):
    print(f"\n🎨 Visualizing Fold {fold} (Full Slide)...")
    
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.0).to(device)
    model_path = os.path.join(MODEL_PATH, f'transmil_fold_{fold}_best.pth')
    
    if not os.path.exists(model_path):
        continue
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_df = df[df[f'fold_{fold}'] == 'test']
    patients_to_process = test_df['patient'].tolist()

    save_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
    os.makedirs(save_dir, exist_ok=True)

    for patient_id in tqdm(patients_to_process, desc=f"Fold {fold}"):
        matching_files = file_map.get(patient_id, [])
        if not matching_files: continue

        true_label = df[df['patient'] == patient_id]['msi'].values[0]
        label_str = "MSI" if true_label == 1 else "MSS"

        # 1) 전체 슬라이드의 모든 패치와 좌표를 로드 (샘플링 안 함!)
        all_features, all_coords = [], []
        for fp in matching_files:
            with h5py.File(fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
                all_coords.append(f["coords"][:])
                
        features_tensor = torch.cat(all_features, dim=0) # 예: 50,000개
        coords_array = np.concatenate(all_coords, axis=0)
        total_patches = features_tensor.shape[0]

        all_logits = []
        all_attn_scores = []

        # 2) 4096개씩 조각(Chunk)내어 모델에 통과시킴 (OOM 방지)
        with torch.no_grad():
            for i in range(0, total_patches, MAX_PATCHES):
                chunk_feat = features_tensor[i : i + MAX_PATCHES]
                current_chunk_size = chunk_feat.shape[0]
                
                # 마지막 조각이 4096개보다 작으면 0으로 패딩하여 형태를 맞춤
                pad_size = 0
                if current_chunk_size < MAX_PATCHES:
                    pad_size = MAX_PATCHES - current_chunk_size
                    pad_tensor = torch.zeros((pad_size, chunk_feat.shape[1]))
                    chunk_feat = torch.cat([chunk_feat, pad_tensor], dim=0)

                features_dict = {'features': chunk_feat.unsqueeze(0).to(device)}
                
                # 모델 추론
                logits, attn_scores = model(features_dict, return_raw_attention=True)
                all_logits.append(logits.item())
                
                # 패딩했던 가짜 어텐션 점수는 잘라내고 진짜만 보존
                chunk_attn = attn_scores.squeeze().cpu().numpy()
                if pad_size > 0:
                    chunk_attn = chunk_attn[:-pad_size]
                    
                all_attn_scores.append(chunk_attn)

        # 3) 조각났던 어텐션 점수들을 원상복구 (전체 슬라이드 점수)
        final_attn_scores = np.concatenate(all_attn_scores) # 다시 50,000개로 복구됨
        
        # 예측 확률은 각 조각의 예측값 평균 사용
        avg_logit = np.mean(all_logits)
        probs = torch.sigmoid(torch.tensor(avg_logit)).item()

        # 4) 시각화 (전체 패치가 모두 표시됨)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor('black')
        fig.patch.set_facecolor('white')

        vmax = np.percentile(final_attn_scores, 99) 
        
        # 5만 개의 점이 모두 찍힘!
        scatter = ax.scatter(
            coords_array[:, 0], -coords_array[:, 1], 
            c=final_attn_scores, cmap='jet', s=10, vmax=vmax, alpha=0.8, edgecolors='none'
        )
        
        plt.colorbar(scatter, ax=ax, label='Attention Score')
        
        is_correct = (probs >= 0.5 and true_label == 1) or (probs < 0.5 and true_label == 0)
        title_obj = plt.title(f"ID: {patient_id} | True: {label_str} | Prob: {probs:.4f}")
        if not is_correct:
            plt.setp(title_obj, color='red') 

        plt.axis('equal')
        plt.axis('off')
        
        save_path = os.path.join(save_dir, f"P{probs:.2f}_{label_str}_{patient_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

print(f"\n✨ 전체 슬라이드 시각화 완료! 결과: '{OUTPUT_DIR}'")