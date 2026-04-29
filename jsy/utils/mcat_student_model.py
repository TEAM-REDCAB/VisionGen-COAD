import torch
import torch.nn as nn
import torch.nn.functional as F

# 필요한 기본 유틸리티 모듈 (논문 원본 구조 반영)
class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.fc = nn.Linear(dim1, dim2)
        self.selu = nn.SELU()
        self.drop = nn.AlphaDropout(dropout)
    def forward(self, x):
        return self.drop(self.selu(self.fc(x)))

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=256, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, n_classes)
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # 요소별 곱셈 (Gating)
        return A, x

# --- 핵심 MCAT 분류 모델 ---
class MCAT_Student(nn.Module):
    def __init__(self, avg_omic_tensor=None, path_input_dim=1536, omic_input_dim=9, d_model=256, dropout=0.25):
        super(MCAT_Student, self).__init__()
        
        # 1. WSI 인코더 (1536 -> 256)
        self.wsi_net = nn.Sequential(
            nn.Linear(path_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 유전체 데이터(1425개)의 빈자리를 채울 쿼리를 파라미터로 등록합니다.
        if avg_omic_tensor is not None:
            self.latent_queries = nn.Parameter(avg_omic_tensor)
        else:
            self.latent_queries = nn.Parameter(torch.randn(1, 1425, d_model))

        # 3. 교차 어텐션 (Genomic-Guided Co-Attention)
        # batch_first=True로 설정하여 (Batch, Seq, Feature) 형태로 편하게 다룹니다.
        self.coattn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, dropout=dropout, batch_first=True)

        # 4. 모달리티별 트랜스포머 및 어텐션 풀링 (압축)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu', batch_first=True)
        self.path_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=d_model, D=d_model, dropout=dropout, n_classes=1)
        
        self.classifier = nn.Linear(d_model, 1) # 이진 분류 (MSI/MSS)

    def forward(self, x_path):
        # x_path: (Batch, N, 1536) / x_omic: (Batch, 1425, 9)
        batch_size = x_path.shape[0]
        
        # 1. 인코딩
        h_path = self.wsi_net(x_path) # (Batch, N, 256)
        # h_omic = self.omic_net(x_omic) # (Batch, 1425, 256)
        h_omic = self.latent_queries.expand(batch_size, -1, -1)

        # 2. 교차 어텐션 (Query: Omic, Key/Value: Path)
        # 반환값: 변환된 피처(h_path_coattn), 어텐션 가중치(A_coattn)
        h_path_coattn, A_coattn = self.coattn(query=h_omic, key=h_path, value=h_path) 
        # h_path_coattn 형태: (Batch, 1425, 256) -> N차원이 1425로 압축됨!

        # 3. Path Feature 압축 (1425개의 시퀀스를 1개의 벡터로)
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, _ = self.path_attention_head(h_path_trans) # A_path: (Batch, 1425, 1)
        A_path = F.softmax(A_path, dim=1).transpose(1, 2)  # (Batch, 1, 1425)
        h_path_bag = torch.bmm(A_path, h_path_trans).squeeze(1) # (Batch, 256)

        # 이미지만으로 결과 내기
        logits = self.classifier(h_path_bag) # (Batch, 1)
        attn_map = torch.bmm(A_path, A_coattn)
        # BCEWithLogitsLoss를 사용할 것이므로 로짓 자체를 반환하는 것이 좋습니다.
        return logits, h_path_bag, attn_map

class BinaryFocalLoss(nn.Module):
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
    