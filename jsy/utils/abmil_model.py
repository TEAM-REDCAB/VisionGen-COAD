import sys
import os
# sys.path.append(os.path.join(os.getcwd(), '../TRIDENT'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from encoders.trident.slide_encoder_models import ABMILSlideEncoder
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
        
        # 💡 [핵심 수정 1] 피처 증류를 위해 Projector(256차원 추출)와 Classifier(최종 1차원 추출)를 분리합니다.
        self.projector = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    # 💡 [핵심 수정 2] 딕셔너리 'x' 대신 순수 텐서 'features'를 입력으로 받습니다.
    def forward(self, features, return_raw_attention=True):
        
        # 입력받은 순수 텐서를 ABMILSlideEncoder가 좋아하는 딕셔너리 형태로 감싸줍니다 (통역 역할)
        enc_input = {'features': features}
        
        # 어텐션 값은 지식 증류(beta Loss)에 필수이므로 항상 추출하는 것이 좋습니다.
        slide_features, attn = self.feature_encoder(enc_input, return_raw_attention=True)
        
        # 768차원 슬라이드 피처를 256차원으로 축소 (Teacher의 256차원 구조와 맞춤)
        path_bag = self.projector(slide_features)
        
        # 최종 로짓 값 추출
        logits = self.classifier(path_bag).squeeze(1)
        
        # 💡 [핵심 수정 3] MCAT Student와 완벽히 동일하게 로짓, 어텐션, 피처 3가지를 모두 반환합니다.
        return logits, path_bag, attn
