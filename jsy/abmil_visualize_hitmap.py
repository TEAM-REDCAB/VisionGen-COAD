import os
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.abmil_model import BinaryClassificationModel
import config as cf

# 1. 설정 및 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_PATH = cf.get_label_path()
MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models')
FEATS_PATH = cf.get_feats_path()
OUTPUT_DIR = os.path.join(cf.get_results_path(), 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(LABEL_PATH)

# [성능 최적화] 모든 파일을 미리 리스팅해서 딕셔너리에 저장 (O(N^2) 방지)
print("📂 파일 목록 스캔 중...")
all_h5_files = [f for f in os.listdir(FEATS_PATH) if f.endswith('.h5')]
file_map = {}
for f in all_h5_files:
    prefix = f[:12] # 환자 ID 또는 슬라이드 ID 추출
    if prefix not in file_map:
        file_map[prefix] = []
    file_map[prefix].append(os.path.join(FEATS_PATH, f))

# 2. 시각화 실행
for fold in range(5):
    print(f"\n🎨 Visualizing Fold {fold}...")
    
    # 모델 로드 (dropout은 추론 시 무시되지만 구조 일치를 위해 추가)
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.25).to(device)
    model_path = os.path.join(MODEL_PATH, f'abmil_fold_{fold}_best.pth')
    
    if not os.path.exists(model_path):
        print(f"⚠️ {model_path} 없음. 스킵.")
        continue
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 해당 Fold의 Test 환자만 추출
    test_df = df[df[f'fold_{fold}'] == 'test']
    patients_to_process = test_df['patient'].tolist()

    save_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
    os.makedirs(save_dir, exist_ok=True)

    for patient_id in tqdm(patients_to_process, desc=f"Fold {fold}"):
        # 매칭되는 파일 가져오기 (미리 만든 file_map 활용)
        matching_files = file_map.get(patient_id, [])
        if not matching_files:
            continue

        true_label = df[df['patient'] == patient_id]['msi'].values[0]
        label_str = "MSI" if true_label == 1 else "MSS"

        # 데이터 로드
        all_features, all_coords = [], []
        for fp in matching_files:
            try:
                with h5py.File(fp, "r") as f:
                    all_features.append(torch.from_numpy(f["features"][:]))
                    all_coords.append(f["coords"][:])
            except Exception as e:
                print(f"❌ 파일 로드 에러 ({patient_id}): {e}")
                continue

        if not all_features: continue

        features_tensor = torch.cat(all_features, dim=0)
        coords_array = np.concatenate(all_coords, axis=0)

        # 모델 추론
        with torch.no_grad():
            # [주의] BinaryClassificationModel의 forward에 return_raw_attention=True 기능이 구현되어 있어야 합니다.
            features_dict = {'features': features_tensor.unsqueeze(0).to(device)}
            logits, attn_scores = model(features_dict, return_raw_attention=True)
            probs = torch.sigmoid(logits).item()
            attn_scores = attn_scores.squeeze().cpu().numpy()

        # 3. 시각화 (개선됨)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 배경을 검은색으로 설정하면 'jet' 컬러맵의 대비가 더 뚜렷해집니다.
        ax.set_facecolor('black')
        fig.patch.set_facecolor('white')

        vmax = np.percentile(attn_scores, 99) # 상위 1% 기준으로 대비 조정
        
        # Y축 반전(-coords)은 WSI 좌표계 특성상 필수입니다.
        scatter = ax.scatter(
            coords_array[:, 0], -coords_array[:, 1], 
            c=attn_scores, cmap='jet', s=12, vmax=vmax, alpha=0.8, edgecolors='none'
        )
        
        plt.colorbar(scatter, ax=ax, label='Attention Score')
        
        # 판정 결과 강조 (정답 여부에 따라 제목 색상 변경 가능)
        is_correct = (probs >= 0.5 and true_label == 1) or (probs < 0.5 and true_label == 0)
        title_obj = plt.title(f"ID: {patient_id} | True: {label_str} | Prob: {probs:.4f}")
        if not is_correct:
            plt.setp(title_obj, color='red') # 틀린 예측은 빨간색 제목

        plt.axis('equal')
        plt.axis('off')
        
        # 파일명 형식: [확률]_[정답]_[환자ID].png
        save_path = os.path.join(save_dir, f"P{probs:.2f}_{label_str}_{patient_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

print(f"\n✨ 모든 시각화 완료! 결과: '{OUTPUT_DIR}'")