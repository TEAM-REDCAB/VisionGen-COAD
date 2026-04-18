import os
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 진행률 표시를 위해 권장
from config import BinaryClassificationModel
import config as cf

# 1. 설정 및 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_PATH = cf.get_label_path()
MODEL_PATH = os.path.join(cf.get_results_path(),'saved_models')
FEATS_PATH = cf.get_features_path()
OUTPUT_DIR = os.path.join(cf.get_results_path(),'visualizations')  # 저장할 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(LABEL_PATH)

# 4. 반복문 실행
for fold in range(5):
    # 2. 모델 로드 (BinaryClassificationModel 정의는 기존과 동일)
    model = BinaryClassificationModel(input_feature_dim=1536).to(device)
    model_path = os.path.join(MODEL_PATH, f'abmil_fold_{fold}_best.pth')
    
    if not os.path.exists(model_path):
        print(f"⚠️ {model_path} 없음. 스킵.")
        continue
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 대상 환자 리스트 추출 (Test Set 전체)
    test_df = df[df[f'fold_{fold}'] == 'test']
    patients_to_process = test_df['patient'].tolist()

    for patient_id in tqdm(patients_to_process):
        # 실제 라벨 및 파일 찾기
        true_label = df[df['patient'] == patient_id]['msi'].values[0]
        label_str = "MSI" if true_label == 1 else "MSS"
        
        all_files = os.listdir(FEATS_PATH)
        matching_files = [os.path.join(FEATS_PATH, f) for f in all_files if f.startswith(patient_id) and f.endswith('.h5')]
        
        if not matching_files:
            continue

        # 데이터 로드
        all_features, all_coords = [], []
        for fp in matching_files:
            with h5py.File(fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
                all_coords.append(f["coords"][:]) 

        features_tensor = torch.cat(all_features, dim=0)
        coords_array = np.concatenate(all_coords, axis=0)

        # 모델 추론
        with torch.no_grad():
            features_dict = {'features': features_tensor.unsqueeze(0).to(device)}
            logits, attn_scores = model(features_dict, return_raw_attention=True)
            probs = torch.sigmoid(logits).item()
            attn_scores = attn_scores.squeeze().cpu().numpy()

        # 시각화 생성
        plt.figure(figsize=(12, 10))
        vmax = np.percentile(attn_scores, 99) 
        
        scatter = plt.scatter(
            coords_array[:, 0], -coords_array[:, 1], 
            c=attn_scores, cmap='jet', s=10, vmax=vmax, alpha=0.7
        )
        
        plt.colorbar(scatter, label='Attention Score')
        plt.title(f"ID: {patient_id} | True: {label_str} | Prob: {probs:.4f}")
        plt.axis('equal')
        plt.axis('off')
        
        # 파일 저장 (파일명에 라벨과 예측 확률을 포함하면 나중에 확인하기 편합니다)
        save_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"P{probs:.2f}_{label_str}_{patient_id}.png")
        plt.savefig(save_path, dpi=150) # 전수 조사용이므로 용량을 위해 dpi를 약간 낮춤
        
        # ⭐ 중요: 메모리 해제를 위해 플롯 닫기
        plt.close()

print(f"✅ 모든 시각화 완료! 결과물은 '{OUTPUT_DIR}' 폴더에서 확인하세요.")