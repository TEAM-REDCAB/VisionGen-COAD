import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score, accuracy_score # AUC, Acc 계산용 추가

def train_loop_classification_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss = 0.

    print('\n')
    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))
    
    # 수정: event_time, c 제거
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)

        # 수정: 분류 모델은 보통 logits, 예측확률(Y_prob), 예측라벨(Y_hat)을 반환한다고 가정
        logits, Y_prob, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        
        # 수정: 일반적인 CrossEntropyLoss 형태
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        # 예측 결과 저장
        all_probs[batch_idx] = Y_prob.detach().cpu().numpy()[0]
        all_preds[batch_idx] = Y_hat.item()
        all_labels[batch_idx] = label.item()

        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, pred: {}'.format(batch_idx, loss_value + loss_reg, label.item(), Y_hat.item()))
        
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # 에포크 단위 손실 및 에러 계산
    train_loss /= len(loader)
    
    # 이진 분류 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    print('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, train_auc: {:.4f}'.format(epoch, train_loss, acc, auc))

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

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in enumerate(loader):

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
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        all_probs[batch_idx] = Y_prob.cpu().numpy()[0]
        all_preds[batch_idx] = Y_hat.item()
        all_labels[batch_idx] = label.item()

        val_loss += loss_value + loss_reg

    val_loss /= len(loader)
    
    acc = accuracy_score(all_labels, all_preds)
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', acc, epoch)
        writer.add_scalar('val/auc', auc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_classification_coattn(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_probs = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader)))
    all_labels = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label) in enumerate(loader):
        
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits, Y_prob, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        prob = Y_prob.cpu().numpy()[0]
        pred = Y_hat.item()
        true_label = label.item()

        all_probs[batch_idx] = prob
        all_preds[batch_idx] = pred
        all_labels[batch_idx] = true_label
        
        # 결과를 딕셔너리로 저장 (개별 환자별 분석용)
        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id), 
                'prob': prob, 
                'pred': pred,
                'label': true_label
            }
        })

    acc = accuracy_score(all_labels, all_preds)
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    return patient_results, acc, auc