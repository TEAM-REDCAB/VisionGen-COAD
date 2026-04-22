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
    def __init__(self, clin_txt_path, mut_csv_path, npy_path, data_dir):
        self.data_dir = data_dir
        
        # 1. 텍스트 파일 로드 (type 기둥과 MSIMUT 정답 강제 하드코딩)
        slide_data = pd.read_csv(clin_txt_path, sep='\s+', low_memory=False)
        slide_data['label'] = slide_data['type'].map({'MSS': 0, 'MSIMUT': 1})
        slide_data = slide_data.dropna(subset=['label']).reset_index(drop=True)
        slide_data['label'] = slide_data['label'].astype(int)
        
        self.slide_data = slide_data
        
        # 2. 유전체 환자목록 로드 (npy 행 인덱스 매칭용)
        mut_df = pd.read_csv(mut_csv_path)
        unique_patients = mut_df['patient_nm'].unique()
        # npy의 row 순서는 unique_patients 리스트 순서와 100% 동일함 (팀원 코드 분석 결과)
        self.patient_to_npy_idx = {pt_nm: i for i, pt_nm in enumerate(unique_patients)}
        # 3. 팀원분이 완성하신 통짜 큐브(.npy) 로드
        self.genomic_matrix = np.load(npy_path)

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        # 1. 환자 정보 (case_id 대신 기둥 이름인 patient 로 변경)
        patient_id = self.slide_data['patient'][idx] # TCGA-XXXX-XX 형태
        label = self.slide_data['label'][idx]
        
        
        # glob 검색: 환자 이름(TCGA-XXXX)으로 시작하는 모든 .h5 슬라이드 싹쓸이 검색!
        wsi_paths = glob.glob(os.path.join(self.data_dir, f'{patient_id}*.h5'))
        
        
       
        # 2. WSI 이미지 텐서 읽어오기(jsy님 폴더에는 pt_files란 중간 폴더가 없으므로 경로 심플하게 수정)
        path_features = []
        for wsi_path in wsi_paths:
            try:
                with h5py.File(wsi_path, 'r') as f:
                    # h5 파일 내부에서 추출된 특징을 꺼냄
                    if 'features' in f:
                        feat = f['features'][:]
                    else:
                        key = list(f.keys())[0]
                        feat = f[key][:]
                path_features.append(torch.tensor(feat, dtype=torch.float32))
            except Exception:
                pass
                
        # 환자 이미지가 1장도 없을 때의 방어 (일단 차원 붕괴를 막기 위해 빈 박스 삽입)
        if len(path_features) == 0:
            path_features.append(torch.zeros((1, 1536), dtype=torch.float32))
        path_features = torch.cat(path_features, dim=0) # 모델에 던지기 직전에 리스트를 하나의 커다란 텐서 벽돌로 압축

        # 3. 유전체 데이터 (1425 x 9) 시퀀스로 추출하기
        npy_idx = self.patient_to_npy_idx.get(patient_id, -1)
        if npy_idx != -1:
            genomic_features = torch.tensor(self.genomic_matrix[npy_idx], dtype=torch.float32)
        else:
            # 돌연변이 기록이 없는 환자는 0으로 채워서 넘깁니다.
            genomic_features = torch.zeros((1425, 9), dtype=torch.float32)
            
        # 4. 모델로 던져줄 3가지 텐서 (배달 완료)
        return path_features, genomic_features, label, patient_id

