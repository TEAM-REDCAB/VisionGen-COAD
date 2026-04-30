import torch
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve, average_precision_score
from tqdm import tqdm
import numpy as np

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
    
    for batch_idx, (data_wsi, data_omic, label) in enumerate(pbar):
        data_wsi = data_wsi.to(device)
        data_omic = data_omic.to(device)
        label = label.type(torch.FloatTensor).to(device)
        
        # # --- 🚨 절제 실험 (Ablation): 유전체 데이터 블라인드 처리 ---
        # # data_omic과 똑같은 크기의 0행렬을 만들어서 모델에 넣습니다.
        # if torch.rand(1).item() < 0.5:
        #     data_omic = torch.zeros_like(data_omic).to(device)        
        logits, *_ = model(data_wsi, data_omic)
        
        
        logits = logits.squeeze(dim=-1)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
            
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        loss = loss / gc 
        loss.backward()
        
        if (batch_idx + 1) % gc == 0 or (batch_idx + 1) == len(loader): 
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss += loss_value
        
        probs = torch.sigmoid(logits)
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
    
    # return avg_loss, auroc, auprc, f1, best_thresh


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
        for data_wsi, data_omic, label in pbar:
            data_wsi = data_wsi.to(device)
            data_omic = data_omic.to(device)
            label = label.type(torch.FloatTensor).to(device)
            
            logits, *_ = model(data_wsi, data_omic)
            
            logits = logits.squeeze(dim=-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
                
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            # preds = (probs > 0.5).float()
            
            all_labels.extend(label.cpu().numpy())
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