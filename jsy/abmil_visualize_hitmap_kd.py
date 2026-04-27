import os
import numpy as np
import pandas as pd
import h5py
import torch
import matplotlib.pyplot as plt
from utils.abmil_model import BinaryClassificationModel
import config as cf

# --- 설정 및 경로 ---
LABEL_PATH = cf.get_label_path()
FEATS_PATH = cf.get_feats_path()
COORDS_PATH = cf.get_coords_path()
# 💡 KD 훈련 모델이 저장된 경로 지정
MODEL_PATH = os.path.join(cf.get_results_path(), 'saved_models_kd')
# 시각화 결과 저장 폴더
VIS_PATH = os.path.join(cf.get_results_path(), 'visualizations')
os.makedirs(VIS_PATH, exist_ok=True)

def plot_patient_attention(patient_id, fold_num=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 환자의 정답(Label) 정보 로드
    df = pd.read_csv(LABEL_PATH)
    patient_info = df[df['patient'] == patient_id]
    if patient_info.empty:
        print(f"⚠️ 환자 {patient_id}를 CSV 파일에서 찾을 수 없습니다.")
        return
    
    true_label = patient_info.iloc[0]['msi']
    true_class = "MSI-H" if true_label == 1 else "MSS"
    
    # 2. 모델 로드 (지정된 Fold의 베스트 모델)
    model = BinaryClassificationModel(input_feature_dim=1536, dropout=0.0).to(device)
    model_file = os.path.join(MODEL_PATH, f'best_model_fold{fold_num}.pt')
    
    if not os.path.exists(model_file):
        print(f"⚠️ 모델 파일이 없습니다: {model_file}")
        return
        
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_thresh = checkpoint['best_thresh']
    model.eval()

    # 3. 환자의 피처 및 좌표 데이터 파일 찾기
    all_files = os.listdir(FEATS_PATH)
    matching_files = [f for f in all_files if f.startswith(patient_id) and f.endswith('.h5')]
    
    if not matching_files:
        print(f"⚠️ 환자 {patient_id}의 .h5 피처 파일을 찾을 수 없습니다.")
        return

    all_features = []
    all_coords = []
    
    for fname in matching_files:
        feat_fp = os.path.join(FEATS_PATH, fname)
        coord_fp = os.path.join(COORDS_PATH, fname.replace('.h5', '_patches.h5'))
        
        with h5py.File(feat_fp, 'r') as f:
            all_features.append(torch.from_numpy(f['features'][:]))
        with h5py.File(coord_fp, 'r') as f:
            all_coords.append(f['coords'][:])

    features = torch.cat(all_features, dim=0).to(device)
    coords = np.concatenate(all_coords, axis=0)

    # 4. 모델 추론 및 어텐션 점수 획득
    with torch.no_grad():
        features_dict = {'features': features}
        # return_raw_attention=True로 설정하여 패치별 중요도(attn) 획득
        logits, attn = model(features_dict, return_raw_attention=True)
        prob = torch.sigmoid(logits).item()
        attn = attn.squeeze().cpu().numpy()

    # 예측 결과 판별
    pred_label = 1 if prob >= best_thresh else 0
    pred_class = "MSI-H" if pred_label == 1 else "MSS"
    
    # 시각화를 극대화하기 위한 어텐션 점수 정규화 (Min-Max)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # 5. 히트맵 시각화 (Scatter Plot)
    plt.figure(figsize=(12, 10))
    
    # 'jet' 컬러맵 사용: 파란색(낮은 중요도) -> 노란색 -> 빨간색(높은 중요도)
    # s는 점의 크기, alpha는 투명도입니다. 슬라이드 크기에 맞춰 조절하세요.
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=attn_norm, cmap='jet', s=10, alpha=0.7)
    
    plt.colorbar(sc, label='Normalized Attention Score')
    
    # WSI 좌표계는 보통 y축이 아래로 증가하므로 뒤집어줍니다.
    plt.gca().invert_yaxis()
    plt.axis('equal') # 슬라이드의 실제 비율 유지
    
    # 타이틀에 중요 정보 표시
    title_str = (f"Patient: {patient_id} | True: {true_class} | Pred: {pred_class}\n"
                 f"Prob: {prob:.4f} (Threshold: {best_thresh:.4f})")
    plt.title(title_str, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # 이미지 저장 및 출력
    save_filename = f"{patient_id}_Fold{fold_num}_True{true_label}_Pred{pred_label}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_PATH, save_filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 시각화 완료! 저장 위치: {os.path.join(VIS_PATH, save_filename)}")

if __name__ == "__main__":
    # 실행 예시: 테스트 셋에 있는 특정 환자 ID와 사용하고자 하는 폴드 번호를 입력합니다.
    # plot_patient_attention('TCGA-A6-5661', fold_num=0) 
    pass