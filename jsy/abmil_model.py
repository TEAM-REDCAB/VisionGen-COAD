import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'TRIDENT'))
from trident.slide_encoder_models import ABMILSlideEncoder
import torch.nn as nn

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
