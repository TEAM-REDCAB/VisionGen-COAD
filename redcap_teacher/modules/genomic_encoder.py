"""
modules/genomic_encoder.py
유전체 행렬 (1425, 9) → (1425, 256) 임베딩
원본: model_2.py의 Genomic_Interpreter + SNN_Block
mcat 폴더 의존성 제거 → SNN_Block 인라인 구현
"""

import torch
import torch.nn as nn


class SNN_Block(nn.Module):
    """
    원본 출처: mcat/model_utils.py
    Self-Normalizing Network Block (ELU + AlphaDropout)
    """
    def __init__(self, dim1: int, dim2: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class GenomicEncoder(nn.Module):
    """
    1425개 돌연변이 행 각각을 임베딩 → 256차원 토큰 시퀀스로 변환

    입력 컬럼 구조 (9열):
        [0]   variant_id   (정수 ID)
        [1]   variant_class_id (정수 ID)
        [2:8] function_ids  (6개 정수 ID)
        [8]   VAF           (0~1 실수)
    """

    def __init__(self, vocab_sizes: dict, out_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        # padding_idx=0 : lhj님 인코딩 규칙 (0번 = 패딩)
        self.emb_var  = nn.Embedding(vocab_sizes['var']  + 1, 128, padding_idx=0)
        self.emb_vc   = nn.Embedding(vocab_sizes['vc']   + 1, 32,  padding_idx=0)
        self.emb_func = nn.Embedding(vocab_sizes['func'] + 1, 32,  padding_idx=0)

        # 128 + 32 + 32 + 1(VAF) = 193 → out_dim
        self.proj = SNN_Block(dim1=193, dim2=out_dim, dropout=dropout)

    def forward(self, x_omic):
        # x_omic : (1425, 9)  float32
        var_id = x_omic[..., 0].long()          # (1425,)
        vc_id  = x_omic[..., 1].long()          # (1425,)
        f_ids  = x_omic[..., 2:8].long()        # (1425, 6)
        vaf    = x_omic[..., 8].unsqueeze(-1)   # (1425, 1)

        h_var  = self.emb_var(var_id)           # (1425, 128)
        h_vc   = self.emb_vc(vc_id)             # (1425, 32)
        h_func = self.emb_func(f_ids)           # (1425, 6, 32)
        h_func = torch.mean(h_func, dim=-2)     # (1425, 32)  순서 없는 기능 → 평균

        h = torch.cat([h_var, h_vc, h_func, vaf], dim=-1)  # (1425, 193)
        return self.proj(h)                                  # (1425, out_dim)
