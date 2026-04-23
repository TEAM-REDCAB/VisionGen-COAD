"""
modules/classifier_head.py
최종 분류 헤드: 256차원 표현 → logits (n_classes,)
원본: model_2.py의 self.classifier
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, h):
        # h      : (256,)
        # return : logits (1, n_classes)
        return self.fc(h).unsqueeze(0)
