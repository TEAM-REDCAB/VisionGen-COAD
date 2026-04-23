"""
modules/coattn_fusion.py
Co-Attention Fusion 모듈 ★ 프로젝트 핵심

path (N_patches, 256) + omic (1425, 256)
    → h_fused (256,) + attention_scores dict

원본: model_2.py의 coattn + path_transformer + omic_transformer
      + path_attention_head + omic_attention_head + path_rho + omic_rho + mm

Attn_Net_Gated 인라인 구현 → mcat 의존성 제거
MultiheadAttention만 mcat/model_coattn.py 사용 (400줄 커스텀 구현체, 그대로 유지)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class Attn_Net_Gated(nn.Module):
    """
    원본 출처: mcat/model_utils.py
    Gated Attention Network: 각 토큰의 중요도 가중치 계산
    """
    def __init__(self, L: int = 256, D: int = 256, dropout: float = 0.25, n_classes: int = 1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(),    nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        # x : (seq_len, L)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)   # element-wise gate
        return A, x                   # A: (seq_len, n_classes), x: (seq_len, L)


class CoAttnFusion(nn.Module):
    """
    MCAT Co-Attention Fusion

    Teacher-Student 증류 포인트:
        - A_coattn  : omic × path 교차 어텐션 행렬  → 증류 대상 1
        - h_fused   : 최종 256차원 표현             → 증류 대상 2 (Hinton KD면 불필요)
    """

    def __init__(
        self,
        embed_dim: int  = 256,
        num_heads: int  = 1,
        dropout: float  = 0.25,
        mcat_path: str  = '/home/team1/cyl/coad_project_train/mcat',
    ):
        super().__init__()

        # MultiheadAttention : mcat 커스텀 구현체 유지 (PyTorch 기본과 반환값 다름)
        sys.path.append(mcat_path)
        from model_coattn import MultiheadAttention
        self.coattn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # ── Path branch ─────────────────────────────────────────────────
        self.path_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=512,
                dropout=dropout, activation='relu'
            ),
            num_layers=2,
        )
        self.path_attn_head = Attn_Net_Gated(L=embed_dim, D=embed_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # ── Omic branch ─────────────────────────────────────────────────
        self.omic_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=512,
                dropout=dropout, activation='relu'
            ),
            num_layers=2,
        )
        self.omic_attn_head = Attn_Net_Gated(L=embed_dim, D=embed_dim, dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        # ── Fusion ───────────────────────────────────────────────────────
        self.mm = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),     nn.ReLU(),
        )

    def forward(self, h_path, h_omic):
        """
        h_path : (N_patches, 256)
        h_omic : (1425, 256)
        return : h_fused (256,), attn_scores dict
        """
        h_path_bag = h_path.unsqueeze(1)   # (N_patches, 1, 256) → seq 차원
        h_omic_bag = h_omic.unsqueeze(1)   # (1425,    1, 256)

        # Co-Attention: Query=omic, Key=Value=path
        # 메모리 킬러 (N_patches×1425 행렬) → grad_checkpoint으로 역전파 재계산
        def _coattn(q, k):
            return self.coattn(q, k, k)

        h_path_coattn, A_coattn = grad_checkpoint(
            _coattn, h_omic_bag, h_path_bag, use_reentrant=False
        )
        # h_path_coattn : (1425, 1, 256)
        # A_coattn      : (1, 1425, N_patches)

        # ── Path branch ─────────────────────────────────────────────────
        h_path_trans   = self.path_transformer(h_path_coattn)             # (1425, 1, 256)
        A_path, h_path_out = self.path_attn_head(h_path_trans.squeeze(1)) # (1425, 1), (1425, 256)
        A_path         = A_path.transpose(1, 0)                           # (1, 1425)
        h_path_out     = torch.mm(F.softmax(A_path, dim=1), h_path_out)   # (1, 256)
        h_path_out     = self.path_rho(h_path_out).squeeze(0)             # (256,)

        # ── Omic branch ─────────────────────────────────────────────────
        h_omic_trans   = self.omic_transformer(h_omic_bag)
        A_omic, h_omic_out = self.omic_attn_head(h_omic_trans.squeeze(1))
        A_omic         = A_omic.transpose(1, 0)
        h_omic_out     = torch.mm(F.softmax(A_omic, dim=1), h_omic_out)
        h_omic_out     = self.omic_rho(h_omic_out).squeeze(0)

        # ── Fusion ───────────────────────────────────────────────────────
        h_fused = self.mm(torch.cat([h_path_out, h_omic_out], dim=0))    # (256,)

        attn_scores = {
            'coattn': A_coattn,   # Teacher-Student 증류 포인트
            'path':   A_path,
            'omic':   A_omic,
        }
        return h_fused, attn_scores
