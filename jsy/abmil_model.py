import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'TRIDENT'))
from trident.slide_encoder_models import ABMILSlideEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42

def get_label_path():
    # 1. 데이터 로드 (환자 ID와 MSI/MSS 타입만 있는 상태)
    df = pd.read_csv('./common_patients.txt', sep='\t')
    # 가정: df는 'patient'와 'type' (0: MSS, 1: MSI) 컬럼을 가짐
    df['msi'] = df['type'].map({'MSS':0, 'MSIMUT':1})

    # 2. 금고에 넣을 Test Set 20%를 완전히 격리
    # stratify=df['type']를 설정하여 Test Set에도 원본 비율과 똑같이 MSI/MSS가 섞이게 함
    df_train_val, df_test = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['msi'], 
        random_state=SEED  # 재현성을 위한 Seed 고정
    )

    # 3. 각 Fold별 컬럼 생성 및 초기화
    for i in range(5):
        df[f'fold_{i}'] = 'none'

    # 4. Train/Val 데이터(80%) 내에서 Stratified 5-Fold 수행
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # skf.split은 원본의 클래스 비율을 유지하며 인덱스를 반환합니다
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['msi'])):
        
        # 쪼개진 인덱스를 원본 데이터프레임의 실제 인덱스와 매핑
        actual_train_idx = df_train_val.iloc[train_idx].index
        actual_val_idx = df_train_val.iloc[val_idx].index
        
        # 해당 Fold 컬럼에 train, val, test 상태 기록
        df.loc[actual_train_idx, f'fold_{fold_idx}'] = 'train'
        df.loc[actual_val_idx, f'fold_{fold_idx}'] = 'val'
        df.loc[df_test.index, f'fold_{fold_idx}'] = 'test'

    # 5. 결과 확인
    os.makedirs('labels', exist_ok=True)
    label_path = os.path.join('labels', f'clinical_data_seed_{SEED}.csv')
    df.to_csv(label_path, index=False)
    return label_path

def get_features_path():
    features_path = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
    os.makedirs(features_path, exist_ok=True)
    return features_path

def get_results_path():
    results_path = os.path.join('./results', f'seed_{SEED}')
    os.makedirs(results_path, exist_ok=True)
    return results_path




class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0.25, gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)
        
        if return_raw_attention:
            return logits, attn
        
        return logits

class H5Dataset(Dataset):
    # [수정됨] fold_col 파라미터 추가하여 동적으로 Fold 컬럼을 바라보도록 변경
    def __init__(self, feats_path, df, split, fold_col='fold_0', num_features=512):
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        
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
        row = self.df.iloc[idx]
        patient_id = row['patient']
        file_paths = self.patient_to_files.get(patient_id, [])
        
        if not file_paths:
            raise FileNotFoundError(f"환자 {patient_id}에 대한 .h5 파일을 찾을 수 없습니다.")

        # 여러 슬라이드의 피처를 리스트에 담은 후 하나로 병합
        all_features = []
        for fp in file_paths:
            with h5py.File(fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
        
        features = torch.cat(all_features, dim=0)

        # 학습 시 고정된 개수로 샘플링
        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))  # Oversampling
            features = features[indices]

        label = torch.tensor(row["msi"], dtype=torch.float32)
        return features, label

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: 양성 클래스(MSI, Target=1)에 대한 가중치 베이스라인
        gamma: 쉬운 예제의 Loss를 얼마나 깎아낼 것인지 결정하는 집중도 파라미터
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. 수치적 안정성을 보장하는 기본 BCE Loss (reduction='none'으로 각 샘플별 Loss 보존)
        # logits과 targets의 차원(Shape)이 완벽히 동일해야 합니다. (예: [batch_size, 1])
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. 로짓을 확률 p로 변환
        probs = torch.sigmoid(logits)
        
        # 3. 정답 클래스에 대한 예측 확률 (p_t) 계산
        # target이 1(MSI)이면 p, 0(MSS)이면 1-p가 됩니다.
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 4. 자율적 난이도 조절 인자 (Modulating Factor)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # 5. 베이스라인 클래스 가중치 적용 (Alpha Factor)
        # target이 1이면 alpha, 0이면 (1-alpha)를 곱합니다.
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 6. 최종 Focal Loss 산출
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        # 7. 차원 축소 (배치 전체의 평균을 낼 것인지 등)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss