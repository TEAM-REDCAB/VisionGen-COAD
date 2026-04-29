import torch
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve, average_precision_score
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# 💡 지식 증류 복합 손실 함수 (KD Loss)
def kd_loss_fn(s_logits, s_path_bag, s_attn, t_logits, t_path_bag, t_attn, labels, task_criterion, alpha=0.5, beta=0.5, gamma=0.01, T=2.0):
    """
    s_logits, t_logits: (Batch, 1) - 아직 시그모이드를 거치지 않은 로짓
    s_attn: (Batch, 1, N) - 학생의 소프트맥스 완료된 어텐션 맵
    t_attn: (Batch, 1, N) - 티처의 소프트맥스 완료된 어텐션 맵
    """
    # 1. Task Loss: 실제 정답과의 오차
    task_loss = task_criterion(s_logits, labels)
    
    # 2. Logit Distillation: KL-Div with Temperature (정석적 KD)
    s_prob = torch.sigmoid(s_logits / T)
    t_prob = torch.sigmoid(t_logits / T)
    
    # [안전장치] t_prob의 차원을 s_prob에 맞춤
    if s_prob.dim() != t_prob.dim():
        t_prob = t_prob.view_as(s_prob)
        
    logit_loss = F.binary_cross_entropy(s_prob, t_prob) * (T**2)
    
    # 3. Attention Distillation: MSE Loss
    if s_attn.dim() != t_attn.dim():
        t_attn = t_attn.view_as(s_attn)
        
    attn_loss = F.mse_loss(s_attn, t_attn)
    
    # 4. Feature (Path Bag) Distillation: MSE Loss
    # [안전장치] t_path_bag의 차원을 s_path_bag에 맞춤
    if s_path_bag.dim() != t_path_bag.dim():
        t_path_bag = t_path_bag.view_as(s_path_bag)
        
    path_loss = F.mse_loss(s_path_bag, t_path_bag)
    
    # 최종 결합
    return task_loss + (alpha * logit_loss) + (beta * attn_loss) + (gamma * path_loss)

# ==========================================
# 5-Fold Cross Validation 실행 루프
# ==========================================


def train_binary(epoch, model, loader, optimizer, loss_fn, gc=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    train_loss = 0.
    all_labels = []
    all_probs = []
    # all_preds = []
    
    optimizer.zero_grad()
    
    # 1. tqdm 래퍼 적용 (leave=False로 설정하여 에폭이 끝나면 진행바가 깔끔하게 사라지도록 함)
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, dynamic_ncols=True)
    
    for batch_idx, (features, coords, labels, t_logits, t_path_bag, t_attn) in enumerate(pbar):
        features = features.to(device)
        label = labels.type(torch.FloatTensor).to(device)
        t_logits = t_logits.to(device)
        t_path_bag = t_path_bag.to(device)
        t_attn = t_attn.to(device)
        
        s_logits, s_path_bag, s_attn = model(features)
        s_logits = s_logits.squeeze(dim=-1)
        if s_logits.dim() == 0:
            s_logits = s_logits.unsqueeze(0)
        
        loss = kd_loss_fn(s_logits, s_path_bag, s_attn, t_logits, t_path_bag, t_attn, label, loss_fn, alpha=0.5, beta=0.5, gamma=0.01)
        loss_value = loss.item()
        
        loss = loss / gc 
        loss.backward()
        
        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader): 
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss += loss_value
        
        probs = torch.sigmoid(s_logits)
        # preds = (probs > 0.5).float()
        
        all_labels.extend(label.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        # all_preds.extend(preds.detach().cpu().numpy())
        
        # 2. 진행바 우측에 현재 배치의 Loss 실시간 업데이트
        pbar.set_postfix({'Loss': f'{loss_value:.4f}'})

    avg_loss = train_loss / len(loader)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    auroc, auprc, best_thresh, f1 = predict_binary(all_probs, all_labels)
    # 에폭 단위 결과 출력
    print(f'====> Epoch: {epoch:02d} | Train Loss: {avg_loss:.4f} | Train AUROC: {auroc:.4f} | Train AUPRC: {auprc:.4f} | Train F1: {f1:.4f} | Train Thresh: {best_thresh:.4f} <====')


def validate_binary(epoch, model, loader, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() 
    
    val_loss = 0.
    all_labels = []
    all_probs = []
    # all_preds = []
    
    # 검증용 tqdm 적용
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Valid]", leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for features, coords, labels in pbar:
            features = features.to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            logits, *_ = model(features)
            
            logits = logits.squeeze(dim=-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
                
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            # preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            # all_preds.extend(preds.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
    avg_loss = val_loss / len(loader)
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    auroc, auprc, best_thresh, f1 = predict_binary(all_probs, all_labels)
    
    # 에폭 단위 결과 출력
    print(f'====> Epoch: {epoch:02d} | Valid Loss: {avg_loss:.4f} | Valid AUROC: {auroc:.4f} | Valid AUPRC: {auprc:.4f} | Valid F1: {f1:.4f} | Valid Thresh: {best_thresh:.4f} <====')
    
    # return avg_loss, auroc, auprc, f1, best_thresh
    return auroc, auprc, best_thresh

def predict_binary(all_probs, all_labels, threshold_method='youden'):
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        
        if threshold_method == 'youden':
            # youden index
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            index = tpr - fpr
        else:
            # 🔥 F1-Score Maximization 적용 구간 🔥
            # thresholds 배열보다 precisions, recalls 배열의 길이가 1 더 깁니다.
            # 분모가 0이 되는 것을 방지하기 위해 1e-8(epsilon) 추가
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            index = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        
        # F1 Score가 최대가 되는 인덱스 추출
        # best_idx = np.argmax(f1_scores)
        best_idx = np.argmax(index)
        best_thresh = thresholds[best_idx]
        
        # 극단적인 임계값(전부 다 0이거나 전부 다 1로 찍는 경우) 방지를 위한 클리핑(Clipping)
        # best_thresh = np.clip(best_thresh, 0.05, 0.95)
    except ValueError:
        auroc = 0.5
        auprc = 0.0
        best_thresh = 0.5

    # 찾아낸 최적의 Threshold로 예측값(preds)을 한 번에 생성
    all_preds = (all_probs >= best_thresh).astype(float)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return auroc, auprc, best_thresh, f1