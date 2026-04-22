import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 42
MAX_PATCHES = 4096      # TransMIL 64x64용으로 고정


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
    features_path = './trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
    os.makedirs(features_path, exist_ok=True)
    return features_path

def get_results_path():
    results_path = os.path.join('./results', f'seed_{SEED}')
    os.makedirs(results_path, exist_ok=True)
    return results_path


# --- [TransMIL 핵심 모듈: PPEG] ---
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_tokens = x[:, 0], x[:, 1:] # [CLS] 분리
        cnn_feat = feat_tokens.transpose(1, 2).view(B, C, H, W)
        
        # 피라미드 구조의 컨볼루션으로 공간 정보 추출
        x = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1) # [CLS] 다시 합침
        return x

# --- [TransMIL 모델 정의] ---
# config.py 내 TransMIL 클래스 수정
class TransMIL(nn.Module):
    def __init__(self, input_feature_dim=1536, n_classes=1, dropout=0.25):
        super().__init__()
        self.hidden_dim = 512
        self.fc = nn.Sequential(nn.Linear(input_feature_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # 가중치를 직접 뽑기 위해 TransformerEncoderLayer 대신 MultiheadAttention 사용 가능하지만
        # 편의상 레이어를 정의하고 훅(Hook)이나 내부 반환을 구현합니다.
        self.layer1 = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.pos_block = PPEG(512)
        
        # 마지막 레이어: 여기서 어텐션을 뽑습니다.
        self.norm = nn.LayerNorm(512)
        self.layer2 = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x, return_attn=False):
        h = self.fc(x)
        B, N, C = h.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # 1. Layer 1 & PPEG
        h = self.layer1(h)
        H = W = int(np.sqrt(N))
        if H * W == N:
            h = self.pos_block(h, H, W)

        # 2. Layer 2 (마지막 레이어에서 어텐션 추출)
        # self_attention: (Query, Key, Value) 모두 h를 넣음
        # attn_weights: (Batch, Target_Len, Source_Len) -> (B, N+1, N+1)
        h_out, attn_weights = self.layer2(h, h, h, need_weights=True)
        
        # 3. [CLS] 토큰의 어텐션만 추출
        # attn_weights[:, 0, 1:] -> [CLS] 토큰이 나머지 패치들을 보는 점수 (1, N)
        cls_attn = attn_weights[:, 0, 1:] 
        
        h = self.norm(h_out[:, 0])
        logits = self.classifier(h)

        if return_attn:
            return logits, cls_attn
        return logits

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=1536, dropout=0.25):
        super().__init__()
        self.feature_encoder = TransMIL(input_feature_dim=input_feature_dim, dropout=dropout)        

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            logits, attn = self.feature_encoder(x['features'], return_attn=True)
            return logits.squeeze(1), attn # attn: (1, N)
        
        logits = self.feature_encoder(x['features'])
        return logits.squeeze(1)

class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, fold_col='fold_0', num_features=MAX_PATCHES):
        self.df = df[df[fold_col] == split].reset_index(drop=True) 
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        
        # 초기화 단계에서 환자별 파일 매핑 (속도 최적화)
        self.patient_to_files = {}
        all_files = os.listdir(feats_path)
        
        for p_id in self.df['patient']:
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

        # 여러 슬라이드 병합
        all_features = []
        for fp in file_paths:
            with h5py.File(fp, "r") as f:
                all_features.append(torch.from_numpy(f["features"][:]))
        
        features = torch.cat(all_features, dim=0)
        num_available = features.shape[0]

        # --- [수정 포인트] 전수 조사를 위한 조건부 샘플링 ---
        if self.split == 'train':
            # 훈련 시: 고정된 개수(4096)로 샘플링하여 학습 효율 및 규제 효과 부여
            if num_available >= self.num_features:
                indices = torch.randperm(num_available)[:self.num_features]
            else:
                # 패치 수가 부족할 때의 Oversampling
                indices = torch.randint(num_available, (self.num_features,))
            
            # 공간 정보 보존을 위해 인덱스 정렬 필수
            indices = torch.sort(indices)[0]
            features = features[indices]
        else:
            # 검증(Val) 및 테스트(Test) 시: 샘플링 없이 전체 패치 반환 (전수 조사)
            # 이때는 4096개보다 훨씬 많으므로, 추론 루프에서 'Chunking' 처리가 필요합니다.
            pass

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