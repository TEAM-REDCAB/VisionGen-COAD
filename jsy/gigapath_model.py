import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42

def get_label_path():
    df = pd.read_csv('./common_patients.txt', sep='\t')
    df['msi'] = df['type'].map({'MSS':0, 'MSIMUT':1})

    df_train_val, df_test = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['msi'], 
        random_state=SEED  
    )

    for i in range(5):
        df[f'fold_{i}'] = 'none'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['msi'])):
        actual_train_idx = df_train_val.iloc[train_idx].index
        actual_val_idx = df_train_val.iloc[val_idx].index
        
        df.loc[actual_train_idx, f'fold_{fold_idx}'] = 'train'
        df.loc[actual_val_idx, f'fold_{fold_idx}'] = 'val'
        df.loc[df_test.index, f'fold_{fold_idx}'] = 'test'

    os.makedirs('labels', exist_ok=True)
    label_path = os.path.join('labels', f'clinical_data_seed_{SEED}.csv')
    df.to_csv(label_path, index=False)
    return label_path

def get_feats_path():
    return '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_gigapath'

def get_coords_path():
    coords_path = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/patches'
    return coords_path

def get_results_path():
    results_path = os.path.join('./results', f'seed_{SEED}')
    os.makedirs(results_path, exist_ok=True)
    return results_path


class H5Dataset(Dataset):
    def __init__(self, split, fold_col='fold_0', num_features=4096):
        df = pd.read_csv(get_label_path())
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = get_feats_path()
        self.coords_path = get_coords_path()
        # GigaPath의 경우 512보다 더 많은 패치를 넣어도 좋습니다. 
        # RTX 3060 12GB 환경에 맞춰 VRAM을 모니터링하며 이 값을 1024~4096 사이로 조절해 보세요.
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

class BinaryFocalLoss(nn.Module):
    # 기존에 작성하신 코드를 그대로 활용합니다.
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss