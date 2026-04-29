import os
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 사용자님의 기존 클래스 및 함수 임포트
from modules.mcat_multimodal_dataset import MSI_Multimodal_Dataset, get_label_path
from modules.mcat_model import MCAT_Binary

# 설정값 로드 (기존 config 활용)
common_patients = "./data/common_patients.txt"
LABEL_PATH = get_label_path(common_patients)
FEATS_PATH = "/home/team/projects/team_REDCAB/team_project/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath"
NPY_PATH = "./data/genomic_input_matrix.npy"
GENOMIC_PKL_PATH = "./data/genomic_encoding_states.pkl"
RESULT_PATH = "./results_msi"
KNOWLEDGE_SAVE_DIR = "./teacher_knowledge"

def export_knowledge():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(KNOWLEDGE_SAVE_DIR, exist_ok=True)
    
    # 5개 폴드를 순회하며 각각의 지식을 추출
    for fold_idx in range(5):
        print(f"\n{'='*20} Fold {fold_idx} 지식 추출 시작 {'='*20}")
        
        # 2. 해당 폴드의 티처 모델(Best) 로드
        model = MCAT_Binary().to(device)
        model_path = os.path.join(RESULT_PATH, f'best_model_fold{fold_idx}.pt')
        
        if not os.path.exists(model_path):
            print(f"⚠️ Fold {fold_idx} 모델 파일이 없습니다. 건너뜁니다.")
            continue
            
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"체크포인트_fold{fold_idx} 딕셔너리에서 'model_state_dict' 로드 완료.")
        else:
            # 혹시 모를 예외 상황(직접 저장된 경우)을 위한 대비
            model.load_state_dict(checkpoint)
            print("직접 저장된 state_dict를 로드했습니다.")
        model.eval() # 추론 모드로 설정 (드롭아웃 등 비활성화) [cite: 1869]

        # 3. 데이터 누수 방지: 해당 폴드에서 'train'인 환자만 필터링 [cite: 1815, 1834]
        # 사용자님의 데이터셋 구조를 활용하여 'train' 데이터만 로드합니다.
        train_dataset = MSI_Multimodal_Dataset(
            csv_path=LABEL_PATH,
            feats_path=FEATS_PATH,
            npy_path=NPY_PATH,
            pkl_path=GENOMIC_PKL_PATH,
            split='train',
            fold_col=f'fold_{fold_idx}'
        )
        # 배치 사이즈 1로 설정하여 환자 한 명씩 꼼꼼하게 받아쓰기
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        fold_knowledge = {}
        fold_omic_list = []

        # 4. 받아쓰기 (Inference) 시작
        with torch.no_grad(): # 역전파 없이 속도 및 VRAM 최적화 
            for i, (data_wsi, data_omic, label) in enumerate(tqdm(loader)):
                # 현재 환자 ID (Dataset 내부의 정보를 바탕으로 획득)
                patient_id = train_dataset.df.iloc[i]['patient']
                
                # 티처 모델의 입을 열어 로짓과 어텐션 가중치 획득 [cite: 1974, 1975]
                logits, h_path_bag, h_omic, attn_map = model(data_wsi.to(device), data_omic.to(device))
                fold_omic_list.append(h_omic.detach().cpu())
                # 우리가 시각화에서 썼던 그 가중 평균 어텐션 (spatial_attn) 계산
                # (1, 1425) @ (1, 1425, N) -> (1, N) -> (N,)
                t_logit = logits.detach().squeeze().cpu().item()    # (Batch, 1) 시그모이드 전의 로짓 값
                t_path_bag = h_path_bag.detach().cpu().numpy()      # (Batch, 256)
                attn_map = attn_map.detach().squeeze(1).cpu().numpy().astype(np.float16) # (N,)

                
                # 환자 ID를 Key로 하여 지식 저장
                fold_knowledge[patient_id] = {
                    't_logits': t_logit,
                    't_path_bag':t_path_bag,
                    't_attention': attn_map
                }

        # 5. 폴드별 지식 파일 저장 (나중에 스튜던트가 읽을 해설지) [cite: 1872]
        save_file = os.path.join(KNOWLEDGE_SAVE_DIR, f'knowledge_fold{fold_idx}_train.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(fold_knowledge, f)
        # 6. 폴드별 '평균 유전체 피처' 계산 및 텐서 저장
        # (Total_Samples, 1425, 256) -> (1, 1425, 256) 으로 평균 계산
        all_omic_tensor = torch.cat(fold_omic_list, dim=0)
        avg_omic = all_omic_tensor.mean(dim=0, keepdim=True)
        
        avg_omic_file = os.path.join(KNOWLEDGE_SAVE_DIR, f'avg_omic_fold{fold_idx}.pt')
        torch.save(avg_omic, avg_omic_file)
            
        print(f"✅ Fold {fold_idx} 지식 저장 완료: {save_file} ({len(fold_knowledge)}명)")

if __name__ == '__main__':
    export_knowledge()