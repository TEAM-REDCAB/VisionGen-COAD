import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
import config as cf

class H5Dataset(Dataset):
    def __init__(self, split, fold_col='fold_0', num_features=4096):
        df = pd.read_csv(cf.get_label_path())
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = cf.get_feats_path()
        self.coords_path = cf.get_coords_path()
        self.num_features = num_features 
        self.split = split
        
        self.patient_to_files = {}
        all_files = os.listdir(self.feats_path)
        
        for p_id in self.df['patient']:
            matching_files = [
                f for f in all_files 
                if f.startswith(p_id) and f.endswith('.h5')
            ]
            self.patient_to_files[p_id] = matching_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient']
        file_names = self.patient_to_files.get(patient_id, [])
        
        if not file_names:
            raise FileNotFoundError(f"환자 {patient_id}에 대한 .h5 파일을 찾을 수 없습니다.")

        all_features = []
        all_coords = []
        
        for fname in file_names:
            feat_fp = os.path.join(self.feats_path, fname)
            coord_fname = fname.replace('.h5', '_patches.h5')
            coord_fp = os.path.join(self.coords_path, coord_fname)
            
            with h5py.File(feat_fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
            
            # 주의: h5 내부의 좌표 배열 Key 이름이 'coords'라고 가정했습니다.
            # 만약 다르다면 이 부분을 실제 Key 이름으로 수정해야 합니다.
            with h5py.File(coord_fp, "r") as f:
                all_coords.append(torch.from_numpy(f["coords"][:]))
        
        features = torch.cat(all_features, dim=0)
        coords = torch.cat(all_coords, dim=0)

        # Train 단계 샘플링 (Feature와 Coords에 동일한 인덱스를 적용하여 매핑 유지)
        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                # indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
                # [개선 1] 시드 고정 해제: 매 에포크마다 무작위로 뽑도록 변경
                indices = torch.randperm(num_available)[:self.num_features]
            else:
                # indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))  # Oversampling
                indices = torch.randint(num_available, (self.num_features,))  # Oversampling
            
            # [개선 2] 뽑힌 인덱스를 정렬하여 패치의 공간적 순서(Topology) 보존
            indices = torch.sort(indices)[0]

            features = features[indices]
            coords = coords[indices]

        label = torch.tensor(row["msi"], dtype=torch.float32)
        return features, coords, label