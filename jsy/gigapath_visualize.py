import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gigapath.classification_head import ClassificationHead
from gigapath_model import H5Dataset, SEED
import gigapath_model as gm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_feature_contribution():
    # 1. 모델 설정 및 로드 (사용자 설정 반영)
    feat_layer_str = "5-11"
    model = ClassificationHead(
        input_dim=1536,
        latent_dim=768,
        feat_layer=feat_layer_str,
        n_classes=1, # Binary
        freeze=False
    ).to(device)
    
    # 베스트 모델 로드 (weights_only=True로 안전하게)
    model_path = os.path.join(gm.get_results_path(), 'saved_models', 'vit_gigapath_fold_0_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 분류기 가중치(W) 추출
    # nn.Sequential(*[nn.Linear(...)]) 구조이므로 첫 번째 레이어 접근
    classifier_weights = model.classifier[0].weight.data.squeeze(0) # Shape: [D_total]
    
    # 3. 데이터 로드 및 계산
    # 테스트 환자 한 명을 예시로 진행
    df = pd.read_csv(get_label_path())
    sample_patient = df[df['fold_0'] == 'test'].iloc[0]['patient']
    
    # .h5 파일 로드 로직 (기존 코드 활용)
   # 테스트셋 로드 (전체 패치 사용)
    test_dataset = H5Dataset(split="test", fold_col=current_fold_col)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        # GigaPath 슬라이드 인코더를 통해 모든 레이어의 임베딩 추출
        # Shape: [L, 12, 768] (all_layer_embed=True 시)
        img_enc = model.slide_encoder.forward(features_tensor.unsqueeze(0).to(device), 
                                              coords_tensor.unsqueeze(0).to(device), 
                                              all_layer_embed=True)
        
        # 5-11번 레이어 추출 및 결합
        feat_layers = [eval(x) for x in feat_layer_str.split("-")]
        selected_enc = [img_enc[i] for i in feat_layers]
        combined_features = torch.cat(selected_enc, dim=-1).squeeze(0) # Shape: [L, D_total]

        # 4. 각 패치별 기여도(Contribution Score) 계산
        # 각 패치 벡터와 분류기 가중치의 요소별 곱(Element-wise product) 후 합산
        contribution_scores = (combined_features * classifier_weights).sum(dim=-1).cpu().numpy()

    # 5. 시각화 (Grad-CAM 스타일)
    plt.figure(figsize=(12, 10))
    plt.gca().set_facecolor('black')
    
    # 기여도 점수 정규화 (0~1 사이로, 혹은 양수/음수 대비)
    # MSI에 기여하는 양수 값만 강조하려면 ReLU를 적용할 수도 있습니다.
    vmax = np.percentile(contribution_scores, 99)
    vmin = np.percentile(contribution_scores, 1)
    
    scatter = plt.scatter(
        coords_array[:, 0], -coords_array[:, 1], 
        c=contribution_scores, cmap='RdBu_r', # MSI(+)는 빨간색, MSS(-)는 파란색
        s=15, alpha=0.8, vmin=vmin, vmax=vmax, edgecolors='none'
    )
    
    plt.colorbar(scatter, label='Logit Contribution (MSI focus)')
    plt.title(f"Feature Contribution Map: {sample_patient}")
    plt.axis('equal')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_feature_contribution()