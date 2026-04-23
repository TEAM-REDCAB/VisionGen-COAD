# Data Preprocessing & Batch Loader
# workflow image에서 ‘2.dataset’에 해당하는 코드
# 데이터 포맷 변환: 디스크 상의 원시 파일(`.npy`, `.pt`, `.csv`)을 파이썬 스크립트가 계산 연산에 사용할 수 있는 숫자 배열(PyTorch Tensor) 형태로 강제 캐스팅(변환)합니다.
# 배치(Batch) 구성 및 반환: 4번 메인 파일(루프)에서 특정 환자의 인덱스를 호출할 때마다(`__getitem__`), 정해진 순서와 규격에 맞게 해당 환자의 WSI 텐서와 유전체 텐서 쌍을 하나로 묶어서 메인 파일 측으로 반환(Return)합니다.

import os
import glob
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# =========================================================================
# [2번 상자] 배달부 / 1branch 버전 (3.0 Verson : 팀원 jsy+lhj님 파일 완벽 호환 패치판)
# =========================================================================
class Pathomic_Classification_Dataset(Dataset):
    def __init__(self, clin_csv_path, mut_csv_path, npy_path, data_dir):
        self.data_dir = data_dir
        
        # 1. 정답지 CSV 로드
        slide_data = pd.read_csv(clin_csv_path)
        
        # 라벨 매핑 (MSS: 0, MSI: 1)
        slide_data['label'] = slide_data['msi_status'].map({'MSS': 0, 'MSI': 1})
        slide_data = slide_data.dropna(subset=['label'])
        
        # 🔥 핵심: case_id 기준으로 중복 제거
        # 이제 len(self.slide_data)는 슬라이드 수가 아니라 '순수 환자 수'가 됩니다.
        self.slide_data = slide_data.drop_duplicates(subset=['case_id']).reset_index(drop=True)
        self.slide_data['label'] = self.slide_data['label'].astype(int)
        
        # 2. 유전체 환자목록 로드 (npy 행 인덱스 매칭용)
        mut_df = pd.read_csv(mut_csv_path)
        # 만약 mut_df의 patient_nm이 12자리가 넘는다면 .str[:12] 처리가 필요할 수 있습니다.
        unique_patients = mut_df['patient_nm'].unique()
        self.patient_to_npy_idx = {pt_nm: i for i, pt_nm in enumerate(unique_patients)}
        
        # 3. 유전체 데이터 로드
        self.genomic_matrix = np.load(npy_path)

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        # 1. 환자 본체 ID 추출
        case_id = self.slide_data['case_id'][idx] 
        label = self.slide_data['label'][idx]
        
        # 2. WSI 이미지 통합 로드
        # glob이 'TCGA-AA-3715'로 시작하는 DX1, DX2 .h5 파일들을 모두 찾아냅니다.
        wsi_paths = glob.glob(os.path.join(self.data_dir, f'{case_id}*.h5'))
        
        path_features = []
        for wsi_path in wsi_paths:
            try:
                with h5py.File(wsi_path, 'r') as f:
                    # 'features' 키가 없으면 첫 번째 키를 가져오는 유연한 로직
                    feat_key = 'features' if 'features' in f else list(f.keys())[0]
                    feat = f[feat_key][:]
                path_features.append(torch.tensor(feat, dtype=torch.float32))
            except Exception:
                pass
                
        # 방어 코드: 이미지가 없으면 빈 텐서라도 반환
        if len(path_features) == 0:
            path_features.append(torch.zeros((1, 1536), dtype=torch.float32))
        
        # 여러 슬라이드(DX1, DX2...)의 특징을 하나로 합침
        path_features = torch.cat(path_features, dim=0) 
        # 💡 이 부분이 핵심입니다. 
        # 3060 환경에서는 패치를 최대 8,000~10,000개로 제한하는 게 안전합니다.
        # if path_features.shape[0] > 10000:
        #         # 랜덤하게 섞어서 만 개를 뽑거나, 그냥 앞에서부터 만 개를 자릅니다.
        #     path_features = path_features[:10000, :]

        # 3. 유전체 데이터 추출
        # npy_idx 매칭 시 case_id를 사용
        npy_idx = self.patient_to_npy_idx.get(case_id, -1)
        if npy_idx != -1:
            genomic_features = torch.tensor(self.genomic_matrix[npy_idx], dtype=torch.float32)
        else:
            genomic_features = torch.zeros((1425, 9), dtype=torch.float32)
            
        return path_features, genomic_features, label, case_id

