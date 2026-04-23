import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm # 추가

def train_loop_classification_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))
    
    # tqdm 적용: 진행 표시줄 생성
    with tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [Train]", unit="batch") as pbar:

        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in pbar:

            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

            logits, Y_prob, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            
            loss = loss_fn(logits, label)
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            all_probs[batch_idx] = Y_prob.detach().cpu().numpy()[0]
            all_preds[batch_idx] = Y_hat.item()
            all_labels[batch_idx] = label.item()

            train_loss += loss_value + loss_reg

            # tqdm 상태바 업데이트 (매 배치마다 실시간 Loss 표시)
            pbar.set_postfix({
                'loss': f'{loss_value + loss_reg:.4f}',
                'pred': int(Y_hat.item()),
                'label': int(label.item())
            })
            
            loss = loss / gc + loss_reg
            loss.backward()

            if (batch_idx + 1) % gc == 0: 
                optimizer.step()
                optimizer.zero_grad()

    # 에포크 통계 계산
    train_loss /= len(loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {acc:.4f}, train_auc: {auc:.4f}')

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', acc, epoch)
        writer.add_scalar('train/auc', auc, epoch)


def validate_classification_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    
    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))

    # 검증 루프에도 tqdm 적용 (선택 사항이나 확인용으로 좋음)
    with tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [Val]", unit="batch", leave=False) as pbar:

        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in pbar:
            # ... (이전과 동일한 데이터 로딩 코드) ...
            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

            with torch.no_grad():
                logits, Y_prob, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            loss = loss_fn(logits, label)
            val_loss += loss.item()

            all_probs[batch_idx] = Y_prob.cpu().numpy()[0]
            all_preds[batch_idx] = Y_hat.item()
            all_labels[batch_idx] = label.item()

    val_loss /= len(loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2 else roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print(f'Val - Epoch: {epoch}, loss: {val_loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}')
    
    # ... (이하 early_stopping 로직 동일) ...
    if early_stopping:
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_minloss_checkpoint.pt"))
        if early_stopping.early_stop: return True
    return False

def summary_classification_coattn(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))
    
    # 환자별 결과를 담을 딕셔너리 (PKL 저장용)
    results_dict = {}

    with tqdm(enumerate(loader), total=len(loader), desc="[Final Eval]", unit="batch", leave=False) as pbar:

        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in pbar:
            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

            with torch.no_grad():
                logits, Y_prob, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            # 결과 저장
            probs = Y_prob.cpu().numpy()[0]
            pred = Y_hat.item()
            target = label.item()
            
            all_probs[batch_idx] = probs
            all_preds[batch_idx] = pred
            all_labels[batch_idx] = target
            
            # 각 샘플(환자)별 상세 결과 기록
            results_dict[batch_idx] = {'prob': probs, 'pred': pred, 'label': target}

    # 최종 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    # 클래스가 2개일 때와 다중 클래스일 때 구분
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    return results_dict, acc, auc