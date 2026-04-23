"""
modules/path_encoder.py
WSI 패치 특징 (N_patches, 1536) → (N_patches, 256) 임베딩
원본: model_2.py의 self.wsi_net
"""

import torch.nn as nn


class PathEncoder(nn.Module):
    def __init__(self, in_dim: int = 1536, out_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_path):
        # x_path : (N_patches, in_dim)
        # return : (N_patches, out_dim)
        return self.net(x_path)
