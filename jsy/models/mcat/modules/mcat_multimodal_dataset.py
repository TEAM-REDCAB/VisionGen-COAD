import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import h5py
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold


SEED = 42

def get_label_path(path):
    # 이미 생성된 정답지가 있으면 그냥 리턴
    label_path = os.path.join('labels', f'clinical_data_seed_{SEED}.csv')
    if label_path:
        return label_path
    # 공통 환자 정답지를 불러와서 msi 라벨을 생성
    df = pd.read_csv(path, sep='\t')
    df['msi'] = df['type'].map({'MSS':0, 'MSIMUT':1})

    # msi 비율에 따라 트레인/테스트 분리(테스트를 확정적으로 분리)
    df_train_val, df_test = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['msi'], 
        random_state=SEED  
    )

    # 각 폴드별로 컬럼 생성
    for i in range(5):
        df[f'fold_{i}'] = 'none'

    # 비율대로 5폴드 훈련/검증 셋 생성
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['msi'])):
        actual_train_idx = df_train_val.iloc[train_idx].index
        actual_val_idx = df_train_val.iloc[val_idx].index
        
        df.loc[actual_train_idx, f'fold_{fold_idx}'] = 'train'
        df.loc[actual_val_idx, f'fold_{fold_idx}'] = 'val'
        df.loc[df_test.index, f'fold_{fold_idx}'] = 'test'

    os.makedirs('labels', exist_ok=True)
    df.to_csv(label_path, index=False)
    return label_path


class MSI_Multimodal_Dataset(Dataset):
    def __init__(self, split, fold_col,  csv_path, feats_path, npy_path, pkl_path):
        # 1. 임상 데이터 로드 (환자 ID와 라벨)
        df = pd.read_csv(csv_path)
        self.df = df[df[fold_col] == split].reset_index(drop=True)
        self.split = split
        # 2. 유전체 NPY 행렬 로드 (Shape: 266, 1425, 9)
        self.genomic_matrix = np.load(npy_path)
        
        # 3. PKL 로드 (환자 순서 매핑)
        with open(pkl_path, 'rb') as f:
            self.encoding_states = pickle.load(f)
            
        # TODO: 266명 환자의 ID 리스트가 담긴 키를 찾아서 할당
        self.patient_list = self.encoding_states['patient_list'] 
        
        self.feats_path = feats_path
        # 초기화 단계에서 환자별 파일 매핑 딕셔너리 생성 (속도 최적화)
        self.patient_to_files = {}
        all_files = os.listdir(feats_path)
        
        for p_id in self.df['patient']:
            # 환자 ID(12자리)로 시작하고 .h5로 끝나는 모든 파일 찾기
            matching_files = [
                os.path.join(feats_path, f) for f in all_files 
                if f.startswith(p_id) and f.endswith('.h5')
            ]
            self.patient_to_files[p_id] = matching_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 환자 ID 및 라벨 가져오기
        row = self.df.iloc[idx]
        patient_id = row['patient']
        file_names = self.patient_to_files.get(patient_id, [])
        
        # ==========================================
        # 2. WSI 이미지 피처 로드 (h5py 사용)
        # ==========================================
        if not file_names:
            raise FileNotFoundError(f"환자 {patient_id}에 대한 .h5 파일을 찾을 수 없습니다.")

        all_features = []
        # all_coords = []
        
        for fname in file_names:
            feat_fp = os.path.join(self.feats_path, fname)
            # coord_fname = fname.replace('.h5', '_patches.h5')
            # coord_fp = os.path.join(self.coords_path, coord_fname)
            
            with h5py.File(feat_fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
            
            # with h5py.File(coord_fp, "r") as f:
            #     all_coords.append(torch.from_numpy(f["coords"][:]))
        
        path_features = torch.cat(all_features, dim=0)
        # coords = torch.cat(all_coords, dim=0)
        
        # 3. 유전체 피처 로드 (Shape: 1425 x 9)
        npy_idx = self.patient_list.index(patient_id) 
        genomic_features = torch.tensor(self.genomic_matrix[npy_idx], dtype=torch.float32)
        label = torch.tensor(row["msi"], dtype=torch.float32)
        
        return path_features, genomic_features, label

