# Computational Graph & Architecture
# workflow image에서 ‘3.model’에 해당하는 코드
# 신경망 연산 그래프 정의: 메인 파일로부터 GPU로 적재된 인풋 텐서를 입력받아, 차원 수를 맞추는 선형 대수 행렬 곱 연산(Embedding, Projection 등)을 수행하는 계층 집합체입니다.
# 특징 벡터 융합 및 분류: 유전체 텐서 행렬과 이미지 텐서 행렬을 교차 곱셈(Co-Attention)하여 상관관계 특징점(Feature)을 추출해내고, 마지막 노드(Classifier)를 거쳐 최종 환자의 분류 확률값(MSI/MSS Logits)과 XAI용 Attention 가중치 행렬 계산까지만 수행한 뒤 통째로 메인 파일에 반환합니다.
# OOM 해결을 위해 코드 수정(model_2)

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')  # PyTorch 내부 deprecated API 경고 억제
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint  # [메모리 최적화] Gradient Checkpointing

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# =========================================================================
# [원본 의존성 연결] 선생님의 새 폴더 구조(mcat, multimodal)에 맞춘 Import 구조
# =========================================================================
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mcat')))
sys.path.append('/home/team1/cyl/coad_project_train/mcat')

# 1. SNN_Block, Attn_Net_Gated
# -> 원본 출처: mcat/model_utils.py
from model_utils import SNN_Block, Attn_Net_Gated

# 2. MultiheadAttention
# -> 원본 출처: mcat/model_coattn.py 맨 밑부분 (내부에 해킹된 파이토치 어텐션 함수 400줄 포함되어 있음)
from model_coattn import MultiheadAttention

# =========================================================================
# [특수 제작] 팀원 전용 유전체 암호 해독기 (Genomic Interpreter)
# =========================================================================
class Genomic_Interpreter(nn.Module):
    def __init__(self, vocab_sizes, out_dim=256, dropout=0.25):
        super(Genomic_Interpreter, self).__init__()
        
        # 팀원분이 0번을 Padding(빈 공간)으로 남겨뒀으므로 padding_idx=0 적용
        # 각각의 ID(글자)를 의미 있는 딥러닝 텐서(숫자 다발)로 번역하는 사전들 생성
        self.emb_var = nn.Embedding(vocab_sizes['var'] + 1, 128, padding_idx=0)
        self.emb_vc  = nn.Embedding(vocab_sizes['vc'] + 1, 32, padding_idx=0)
        self.emb_func = nn.Embedding(vocab_sizes['func'] + 1, 32, padding_idx=0)
        
        # 번역된 벡터들을 다 합친 길이 = 128 + 32 + 32 + 1(VAF) = 193
        total_in_dim = 128 + 32 + 32 + 1
        
        # 합친 193차원을 MCAT의 공용 규격인 256차원으로 펌핑
        self.proj = SNN_Block(dim1=total_in_dim, dim2=out_dim, dropout=dropout)

    def forward(self, x_omic):
        # x_omic 형태: (1425, 9) (실수형 Float 상태)
        
        # 1. 9칸을 쪼개서 용도에 맞게 정수(Long)로 캐스팅
        var_id = x_omic[..., 0].long()
        vc_id  = x_omic[..., 1].long()
        f_ids  = x_omic[..., 2:8].long() # 6개의 기능 서명
        vaf    = x_omic[..., 8].unsqueeze(-1) # 실제 수학적 수치(비율)이므로 그대로 둠
        
        # 2. 사전(해독기)에 넣어서 고차원 벡터로 변환
        h_var = self.emb_var(var_id)     # 결과: (... 128)
        h_vc  = self.emb_vc(vc_id)       # 결과: (... 32)
        h_func = self.emb_func(f_ids)    # 결과: (... 6, 32)
        
        # 3. 팀원 요구사항 반영: 6개의 기능은 순서가 없으므로 평균(Mean)을 내서 하나로 압축
        h_func_mean = torch.mean(h_func, dim=-2) # 결과: (... 32)
        
        # 4. 해독된 조각들과 원래 수치(VAF)를 이어 붙이기 (드디어 해석 가능한 데이터 완성!)
        h_fused = torch.cat([h_var, h_vc, h_func_mean, vaf], dim=-1) # 결과: (... 193)
        
        # 5. 256 차원으로 변환하여 방출
        h_out = self.proj(h_fused)
        return h_out


# =========================================================================
# [3번 상자] 요리사 / 진정한 REDCAB_MCAT 모델 (해독기 탑재 완료)
# =========================================================================
class MCAT_Single_Branch_Model(nn.Module):
    def __init__(self, vocab_sizes, path_dim=1536, n_classes=2, dropout=0.25):
        super(MCAT_Single_Branch_Model, self).__init__()
        print("[알림] REDCAB_MCAT 구조 활성화: 해독기(Interpreter) 기반 Co-Attention 장착 완료!")
        
        self.n_classes = n_classes
        size = [path_dim, 256, 256] 
        
        # 1. 병리 이미지 임베딩
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        self.wsi_net = nn.Sequential(*fc)
        
        # 2. 유전체(1425 다발) 맞춤형 해독 파이프라인 (직접 짠 Interpreter 활용)
        self.omic_net = Genomic_Interpreter(vocab_sizes=vocab_sizes, out_dim=256)

        # 3. [논문의 상징] Co-Attention 모듈
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)
        
        # 4. 이미지 트랜스포머 및 Gated Attention 모듈
        self.path_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu'), 
            num_layers=2
        )
        self.path_attention_head = Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout))
        
        # 5. 유전체 트랜스포머 및 Gated Attention 모듈
        self.omic_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu'), 
            num_layers=2
        )
        self.omic_attention_head = Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout))
        
        # 6. 결합(Fusion) 및 최종 분류기
        self.mm = nn.Sequential(nn.Linear(256*2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x_path, x_omic):
        # x_path: (N_patches, 1536), x_omic: (1, 1425, 9)

        # [체크포인트 1] wsi_net: N_patches×256 중간값 저장 안 함
        h_path_bag = grad_checkpoint(self.wsi_net, x_path, use_reentrant=False).unsqueeze(1)

        if x_omic.dim() == 3 and x_omic.shape[0] == 1:
            x_omic = x_omic.squeeze(0)

        # [해독기 작동] 1425x9 → 1425x256
        h_omic_bag = self.omic_net(x_omic).unsqueeze(1)

        # [체크포인트 2] Co-Attention: (1, 1425, N_patches) 행렬이 진짜 메모리 킬러!
        # N_patches=20000이면 114MB → 역전파 저장까지 합치면 300MB+ 이상
        # checkpoint로 역전파 시 재계산 → 저장 안 함
        def _run_coattn(q, k):
            return self.coattn(q, k, k)

        h_path_coattn, A_coattn = grad_checkpoint(_run_coattn, h_omic_bag, h_path_bag, use_reentrant=False)

        # 이후 Transformer는 1425 토큰만 처리 (N_patches 무관, 메모리 고정)
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze(0)

        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze(0)

        h = self.mm(torch.cat([h_path, h_omic], axis=0))
        logits = self.classifier(h).unsqueeze(0)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        return logits, Y_hat, attention_scores
