import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import h5py
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import config as cf

class H5Dataset(Dataset):
    def __init__(self, split, fold_col='fold_0', kd_path=None): # num_features 삭제
        df = pd.read_csv(cf.get_label_path())
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = cf.get_feats_path()
        self.coords_path = cf.get_coords_path()
        self.split = split
        
        self.patient_to_files = {}
        all_files = os.listdir(self.feats_path)
        for p_id in self.df['patient']:
            matching_files = [f for f in all_files if f.startswith(p_id) and f.endswith('.h5')]
            self.patient_to_files[p_id] = matching_files

        self.kd_dict = None
        if kd_path and os.path.exists(kd_path):
            with open(kd_path, 'rb') as f:
                self.kd_dict = pickle.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient']
        file_names = self.patient_to_files.get(patient_id, [])
        
        all_features, all_coords = [], []
        for fname in file_names:
            feat_fp = os.path.join(self.feats_path, fname)
            coord_fp = os.path.join(self.coords_path, fname.replace('.h5', '_patches.h5'))
            with h5py.File(feat_fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
            with h5py.File(coord_fp, "r") as f:
                all_coords.append(torch.from_numpy(f["coords"][:]))
        
        # 💡 샘플링 없이 모든 패치를 그대로 사용합니다!
        features = torch.cat(all_features, dim=0)
        coords = torch.cat(all_coords, dim=0)

        # 티처 지식 불러오기
        if self.kd_dict is not None:
            kd_data = self.kd_dict.get(patient_id)
            if kd_data:
                t_logit = torch.tensor(kd_data['t_logits'], dtype=torch.float32)
                # 혹시 남아있을지 모를 배치 차원(1, 256)을 제거하여 (256,) 형태의 1D 벡터로 깔끔하게 펴줍니다.
                t_path_bag = torch.tensor(kd_data['t_path_bag'], dtype=torch.float32).squeeze()
                t_attn = torch.tensor(kd_data['t_attention'], dtype=torch.float32)
                
                # 전체 패치에 대한 어텐션 합이 1이 되도록 재정규화
                t_attn = t_attn / (t_attn.sum() + 1e-8)
            else:
                # KD 데이터가 누락되었을 때 텐서 규격을 맞춰주기 위한 방어 코드
                t_logit = torch.tensor(0.0)
                t_path_bag = torch.zeros(256, dtype=torch.float32) # d_model 크기(256)에 맞춘 빈 텐서
                t_attn = torch.zeros(features.shape[0])
            
            labels = torch.tensor(row["msi"], dtype=torch.float32)
            
            # 💡 리턴값에 t_path_bag 추가 (훈련 루프에서 받는 순서와 동일하게 맞춤)
            return features, coords, labels, t_logit, t_path_bag, t_attn
        else:
            labels = torch.tensor(row["msi"], dtype=torch.float32)
            return features, coords, labels