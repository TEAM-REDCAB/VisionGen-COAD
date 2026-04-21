# workflow image에서 ‘2.dataset’에 해당하는 코드

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# =========================================================================
# [2번 상자] npy 입양 호환
# =========================================================================
class Pathomic_Classification_Dataset(Dataset):
    def __init__(self, clin_csv_path, mut_csv_path, npy_path, data_dir, label_col='msi_status', label_dict={'MSS': 0, 'MSI-H': 1}):
        self.data_dir = data_dir
        
        # 1. 기존 임상 데이터 로드 (WSI 폴더 이름 및 라벨 매칭용)
        slide_data = pd.read_csv(clin_csv_path, low_memory=False)
        slide_data['label'] = slide_data[label_col].map(label_dict)
        slide_data = slide_data.dropna(subset=['label']).reset_index(drop=True)
        slide_data['label'] = slide_data['label'].astype(int)
        
        self.patient_dict = {row['case_id']: [row['slide_id']] for _, row in slide_data.iterrows()}
        self.slide_data = slide_data
        
        # 2. 돌연변이 데이터 환자목록 로드 (npy 행 인덱스 매칭용)
        mut_df = pd.read_csv(mut_csv_path)
        unique_patients = mut_df['patient_nm'].unique()
        # npy의 row 순서는 unique_patients 리스트 순서와 100% 동일함 (팀원 코드 분석 결과)
        self.patient_to_npy_idx = {pt_nm: i for i, pt_nm in enumerate(unique_patients)}
        
        # 3. .npy 로드
        self.genomic_matrix = np.load(npy_path)

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        # 1. 환자 정보
        case_id = self.slide_data['case_id'][idx] # TCGA-XXXX-XX 형태
        label = self.slide_data['label'][idx]
        slide_ids = self.patient_dict[case_id]
        
        # 2. WSI 이미지 텐서 읽어오기
        path_features = []
        for slide_id in slide_ids:
            wsi_path = os.path.join(self.data_dir, 'pt_files', f'{slide_id}.pt')
            try:
                path_features.append(torch.load(wsi_path))
            except Exception:
                # 패치된 파일이 없을 경우 방어 코드
                path_features.append(torch.zeros(1, 1024)) 
        path_features = torch.cat(path_features, dim=0)

        # 3. 유전체 데이터 (1425 x 9) 시퀀스로 흡수하기
        npy_idx = self.patient_to_npy_idx.get(case_id, -1)
        if npy_idx != -1:
            genomic_features = torch.tensor(self.genomic_matrix[npy_idx], dtype=torch.float32)
        else:
            # 돌연변이 기록이 없는 환자는 0으로 채움
            genomic_features = torch.zeros((1425, 9), dtype=torch.float32)
            
        # 4. 모델로 진행할 3가지 텐서
        return path_features, genomic_features, label
