# workflow image에서 ‘3.model’에 해당하는 코드
import torch
import torch.nn as nn

# =========================================================================
# [3번 상자] Multimodal
# =========================================================================
class Pathomic_Single_Array_Model(nn.Module):
    def __init__(self, path_dim=1024, seq_dim=9, n_classes=2):
        super(Pathomic_Single_Array_Model, self).__init__()
        
        # 1. 이미지 임베딩 (N, 1024 -> N, 256)
        self.path_net = nn.Sequential(
            nn.Linear(path_dim, 256), nn.ReLU(), nn.Dropout(0.25)
        )
        
        # 2. 유전체 시퀀스
        # 팀원분이 9개의 칸을 가진 시퀀스로 만들었으므로 입력 통로를 9개로 엽니다.
        # 이를 통과하면 (1425, 9) 의 숫자들이 (1425, 256) 크기로 변환됩니다.
        self.omic_net = nn.Sequential(
            nn.Linear(seq_dim, 256), nn.ReLU(), nn.Dropout(0.25)
        )
        
        # 3. 융합 및 최종 분류 네트워크 (256+256 = 512짜리)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 256), nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, path_features, genomic_features):
        # -----------------------------------------------
        # 1. 이미지 처리 (병합) -> (Batch, 256)
        # -----------------------------------------------
        h_path = torch.mean(path_features, dim=1)
        h_path = self.path_net(h_path)
        
        # -----------------------------------------------
        # 2. 유전체 처리 (Deep-Set Average 추출 기법 적용)
        # -----------------------------------------------
        # genomic_features 모양 : (Batch, 1425, 9)
        # 돌연변이 하나하나를 모두 256차원 딥러닝 특징으로 각자 변환시킵니다.
        h_omic = self.omic_net(genomic_features) # 결과: (Batch, 1425, 256)
        
        # 그 다음 1425개의 돌연변이 특징을 평균(Mean Pooling)을 냅니다.
        # 이렇게 하면 순서 제약이 없어지고 한 환자의 종합 돌연변이 점수가 추출됩니다.
        h_omic = torch.mean(h_omic, dim=1) # 결과: (Batch, 256)
        
        # -----------------------------------------------
        # 3. 데이터 결합 (Concatenation)
        # -----------------------------------------------
        h_fused = torch.cat([h_path, h_omic], dim=1) # 256 + 256 = (Batch, 512)
        
        # 4. 최종 MSI/MSS 예측 확률 도출
        logits = self.classifier(h_fused)
        
        return logits, None, None
