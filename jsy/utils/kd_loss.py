import torch.nn.functional as F

def kd_loss_fn(s_logits, s_attn, t_logits, t_attn, labels, task_criterion, alpha=0.5, beta=0.5):
    """
    s_logits, s_attn: 스튜던트(ABMIL)의 출력
    t_logits, t_attn: 티처(MCAT)의 해설지 (pkl에서 로드)
    labels: 실제 정답 (0 또는 1)
    task_criterion: 기존에 쓰던 BinaryFocalLoss 객체
    """
    # 1. Task Loss: 실제 정답과의 오차 (기존 베이스라인과 동일)
    task_loss = task_criterion(s_logits, labels)
    
    # 2. Logit Loss: 티처의 확신도 모방 (MSE 사용)
    # 티처가 "이건 0.8 정도로 MSI야"라고 한 그 뉘앙스를 배움
    logit_loss = F.mse_loss(s_logits, t_logits)
    
    # 3. Attention Loss: 티처가 주목한 구역 모방 (KL-Divergence)
    # s_attn과 t_attn의 차원(Shape)이 (1, N)으로 동일하다고 가정
    # 스튜던트 어텐션은 log_softmax 처리, 티처 어텐션은 이미 합이 1인 확률 분포여야 함
    s_attn_log = F.log_softmax(s_attn, dim=-1)
    
    # 모델 내부 구조에 따라 차원 맞추기 (t_attn을 s_attn_log와 동일한 차원으로 조절)
    if t_attn.dim() == 1:
        t_attn = t_attn.unsqueeze(0)
        
    attn_loss = F.kl_div(s_attn_log, t_attn, reduction='batchmean')
    
    # 최종 손실: 원래 목표 + (티처 확률 모방) + (티처 시선 모방)
    return task_loss + (alpha * logit_loss) + (beta * attn_loss)