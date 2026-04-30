import os
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.abmil_model import BinaryClassificationModel
from utils.mcat_student_model import MCAT_Student
import config as cf

# ==========================================
# 1. 설정 및 경로
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_PATH = cf.get_label_path()
MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models_abmil_kd')
FEATS_PATH = cf.get_feats_path()

# 💡 앙상블 전용 저장 폴더 생성
OUTPUT_DIR = os.path.join(cf.get_results_path(), 'visualizations_ensemble')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 💡 직전 앙상블 테스트에서 도출된 최적 Threshold 적용
ENSEMBLE_THRESH = 0.2300

df = pd.read_csv(LABEL_PATH)

# [성능 최적화] 모든 파일을 미리 리스팅해서 딕셔너리에 저장
print("📂 파일 목록 스캔 중...")
all_h5_files = [f for f in os.listdir(FEATS_PATH) if f.endswith('.h5')]
file_map = {}
pid_len = len(df['patient'][0])
for f in all_h5_files:
    prefix = f[:pid_len] 
    if prefix not in file_map:
        file_map[prefix] = []
    file_map[prefix].append(os.path.join(FEATS_PATH, f))

# ==========================================
# 2. 5개의 Fold 모델 한 번에 모두 로드
# ==========================================
print("\n🧠 5-Fold 앙상블 모델 로드 중...")
ensemble_models = []
for fold in range(5):
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.0).to(device)
    # model = MCAT_Student().to(device)
    model_path = os.path.join(MODEL_PATH, f'best_model_fold{fold}.pt')
    
    if not os.path.exists(model_path):
        print(f"⚠️ 경고: {model_path} 파일이 없습니다.")
        continue
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    ensemble_models.append(model)

print(f"✅ 총 {len(ensemble_models)}개의 모델 로드 완료!")

# ==========================================
# 3. 앙상블 시각화 실행
# ==========================================
# 외부 코호트(CPTAC)는 fold_0 기준으로 test인 것을 가져오면 전체 환자가 추출됩니다.
test_df = df[df['fold_0'] == 'test']
patients_to_process = test_df['patient'].tolist()

for patient_id in tqdm(patients_to_process, desc="🎨 Generating Ensemble Maps"):
    matching_files = file_map.get(patient_id, [])
    if not matching_files:
        continue

    true_label = df[df['patient'] == patient_id]['msi'].values[0]
    label_str = "MSI" if true_label == 1 else "MSS"

    # H5 피처 데이터 로드
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

    # 💡 5개 모델의 확률과 어텐션을 누적할 변수
    sum_probs = 0.0
    sum_attn_scores = np.zeros(coords_array.shape[0])

    # 개별 모델 추론 및 합산
    with torch.no_grad():
        for model in ensemble_models:
            features_input = features_tensor.unsqueeze(0).to(device)
            logits, _, attn_scores = model(features_input)
            
            # 확률 합산 (Soft Voting)
            prob = torch.sigmoid(logits).item()
            sum_probs += prob
            
            # 어텐션 스코어 합산
            sum_attn_scores += attn_scores.squeeze().cpu().numpy()

    # 💡 최종 앙상블 평균 계산
    ensemble_prob = sum_probs / len(ensemble_models)
    ensemble_attn = sum_attn_scores / len(ensemble_models)

    # ==========================================
    # 4. 이미지 렌더링 및 저장
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')

    # # 상위 1% 기준으로 대비 조정 (5개 모델이 합의한 가장 뜨거운 영역 강조)
    # vmax = np.percentile(ensemble_attn, 99) 
    
    # scatter = ax.scatter(
    #     coords_array[:, 0], -coords_array[:, 1], 
    #     c=ensemble_attn, cmap='jet', s=12, vmax=vmax, alpha=0.8, edgecolors='none'
    # )
    # 💡 수정 후: Z-Score (표준화) 적용
    mean_val = ensemble_attn.mean()
    std_val = ensemble_attn.std()

    # 평균은 0, 표준편차 단위로 변환
    z_scores = (ensemble_attn - mean_val) / (std_val + 1e-8)

    # vmin=0 (평균 이하), vmax=3 (상위 0.1% 극단값)으로 고정
    scatter = ax.scatter(
        coords_array[:, 0], -coords_array[:, 1], 
        c=z_scores, cmap='jet', s=12, vmin=0, vmax=3, alpha=0.8, edgecolors='none'
    )
    
    plt.colorbar(scatter, ax=ax, label='Ensemble Attention Score')
    
    # 💡 앙상블 임계값(0.2460)을 기준으로 정답 여부 판정
    is_correct = (ensemble_prob >= ENSEMBLE_THRESH and true_label == 1) or (ensemble_prob < ENSEMBLE_THRESH and true_label == 0)
    
    title_obj = plt.title(f"ID: {patient_id} | True: {label_str} | Prob: {ensemble_prob:.4f}")
    if not is_correct:
        plt.setp(title_obj, color='red') 

    plt.axis('equal')
    plt.axis('off')
    
    # 파일명 형식: P[확률]_[정답]_[환자ID].png
    pred_text = "MSI" if ensemble_prob >= ENSEMBLE_THRESH else "MSS"
    save_path = os.path.join(OUTPUT_DIR, f"P{ensemble_prob:.2f}_{label_str}_{patient_id}.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"\n✨ 앙상블 어텐션 맵 시각화 완료! 결과 폴더: '{OUTPUT_DIR}'")