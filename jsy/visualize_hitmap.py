import sys
import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 진행률 표시를 위해 권장

# 1. 환경 설정 및 경로 추가
sys.path.append(os.path.join(os.getcwd(), 'TRIDENT'))
from trident.slide_encoder_models import ABMILSlideEncoder

# 1. 설정 및 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = './clinical_data_folds.csv'
MODEL_DIR = './saved_models'
FEATS_PATH = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
OUTPUT_DIR = './visualizations/fold_4'  # 저장할 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 모델 클래스 정의 (저장된 가중치를 불러오기 위해 필수)
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)
        
        if return_raw_attention:
            return logits, attn
        return logits

# 3. 데이터 및 모델 로드
df = pd.read_csv(CSV_PATH)

# 2. 모델 로드 (BinaryClassificationModel 정의는 기존과 동일)
model = BinaryClassificationModel(input_feature_dim=1536).to(device)
model_path = './saved_models/abmil_fold_2_best.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3. 대상 환자 리스트 추출 (Test Set 전체)
test_df = df[df['fold_0'] == 'test']
patients_to_process = test_df['patient'].tolist()

print(f"🚀 총 {len(patients_to_process)}명의 환자에 대해 시각화를 시작합니다.")

# 4. 반복문 실행
for patient_id in tqdm(patients_to_process):
    # 실제 라벨 및 파일 찾기
    true_label = df[df['patient'] == patient_id]['type'].values[0]
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
    save_path = os.path.join(OUTPUT_DIR, f"P{probs:.2f}_{label_str}_{patient_id}.png")
    plt.savefig(save_path, dpi=150) # 전수 조사용이므로 용량을 위해 dpi를 약간 낮춤
    
    # ⭐ 중요: 메모리 해제를 위해 플롯 닫기
    plt.close()

print(f"✅ 모든 시각화 완료! 결과물은 '{OUTPUT_DIR}' 폴더에서 확인하세요.")