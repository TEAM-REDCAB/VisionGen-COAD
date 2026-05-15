import torch
import torch.nn as nn
import torch.nn.functional as F


class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.fc   = nn.Linear(dim1, dim2)
        self.selu = nn.SELU()
        self.drop = nn.AlphaDropout(dropout)

    def forward(self, x):
        return self.drop(self.selu(self.fc(x)))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=256, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(),    nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class MCAT_Student(nn.Module):
    def __init__(
        self,
        avg_omic_tensor=None,
        path_input_dim=1536,
        omic_input_dim=9,
        d_model=256,
        dropout=0.25
    ):
        super().__init__()

        # ── 1. WSI 인코더 (1536 → 256) ──────────────────────────────────────
        self.wsi_net = nn.Sequential(
            nn.Linear(path_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── 2. 학습 가능한 잠재 유전체 쿼리 ──────────────────────────────────
        # avg_omic_tensor: Teacher가 인코딩한 유전체 임베딩의 평균값 (1, 1425, 256)
        # 이 값으로 초기화하면 Teacher의 유전체 표현 공간 근처에서 학습 시작.
        # clone().detach()로 계산 그래프 분리 후 Parameter로 등록.
        if avg_omic_tensor is not None:
            self.latent_queries = nn.Parameter(avg_omic_tensor.clone().detach())
        else:
            self.latent_queries = nn.Parameter(torch.randn(1, 1425, d_model))

        # ── 3. 교차 어텐션 (Genomic-Guided Co-Attention) ────────────────────
        self.coattn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=1,
            dropout=dropout, batch_first=True
        )

        # ── 4. Path 트랜스포머 + 어텐션 풀링 ────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.path_transformer    = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=d_model, D=d_model, dropout=dropout, n_classes=1)

        # ── 5. 분류기 ────────────────────────────────────────────────────────
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x_path):
        """
        x_path : (B, N, 1536)
        returns:
            logits    : (B, 1)
            h_path_bag: (B, 256)
            attn_map  : (B, 1, N)  ← A_path @ A_coattn, Teacher와 동일한 형태
        """
        batch_size = x_path.shape[0]

        # 1. WSI 인코딩
        h_path = self.wsi_net(x_path)                          # (B, N, 256)

        # 2. 잠재 쿼리 배치 복제
        h_omic = self.latent_queries.expand(batch_size, -1, -1) # (B, 1425, 256)

        # 3. Co-Attention : Query=Omic(잠재), Key/Value=Path
        # h_path_coattn: (B, 1425, 256)  A_coattn: (B, 1425, N)
        h_path_coattn, A_coattn = self.coattn(
            query=h_omic, key=h_path, value=h_path
        )

        # 4. Transformer 인코딩
        h_path_trans = self.path_transformer(h_path_coattn)    # (B, 1425, 256)

        # 5. Attention Pooling → 256-dim bag vector
        A_path, _ = self.path_attention_head(h_path_trans)     # (B, 1425, 1)
        A_path    = F.softmax(A_path, dim=1).transpose(1, 2)   # (B, 1, 1425)
        h_path_bag = torch.bmm(A_path, h_path_trans).squeeze(1) # (B, 256)

        # 6. 분류
        logits   = self.classifier(h_path_bag)                  # (B, 1)

        # 7. 복합 어텐션 맵 : (B, 1, 1425) @ (B, 1425, N) → (B, 1, N)
        # Teacher의 t_attn과 동일한 shape으로 KL-Div 계산에 사용됨
        attn_map = torch.bmm(A_path, A_coattn)                  # (B, 1, N)

        return logits, h_path_bag, attn_map


class BinaryFocalLoss(nn.Module):
    """
    클래스 불균형 대응용 Focal Loss.
    alpha : 양성 클래스(MSI)에 부여할 가중치 (0.5~0.75 권장)
    gamma : 쉬운 샘플 억제 강도 (2.0 권장)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss        = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs           = torch.sigmoid(logits)
        p_t             = probs * targets + (1 - probs) * (1 - targets)
        modulating      = (1.0 - p_t) ** self.gamma
        alpha_factor    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss      = alpha_factor * modulating * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
