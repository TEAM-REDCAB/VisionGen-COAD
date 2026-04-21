from argparse import Namespace
from collections import OrderedDict
import os
import pickle 

import numpy as np

import torch
import torch.nn as nn # 추가: 손실 함수용

from datasets.dataset_classification import save_splits
from models.model_genomic import SNN
# 수정: 분류용 모델 임포트 (MCAT_Surv -> MCAT_Classifier)
from models.model_coattn import MCAT_Classifier
from utils.utils import *

from utils.coattn_train_utils import *
from utils.cluster_train_utils import *

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


# 수정: Monitor_CIndex를 분류 평가지표(AUC/Acc)를 추적하는 범용 Monitor_Metric으로 변경
class Monitor_Metric:
    """Saves the model based on a metric where higher is better (e.g., AUC, Accuracy)."""
    def __init__(self):
        self.best_score = None

    def __call__(self, val_metric, model, ckpt_name:str='checkpoint.pt'):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    # 수정: Survival Loss 대신 분류용 CrossEntropyLoss 적용
    if args.task_type == 'classification':
        if args.bag_loss == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type =='snn':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == 'mcat':
        # 수정: MCAT_Classifier로 변경
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Classifier(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation Metric Monitor...', end=' ')
    monitor_metric = Monitor_Metric()
    print('Done!')

    for epoch in range(args.max_epochs):
        # 수정: 분류 루프 실행
        if args.task_type == 'classification':
            if args.mode == 'coattn':
                train_loop_classification_coattn(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_classification_coattn(cur, epoch, model, val_loader, args.n_classes, early_stopping=early_stopping, writer=writer, loss_fn=loss_fn, reg_fn=reg_fn, lambda_reg=args.lambda_reg, results_dir=args.results_dir)
            else:
                raise NotImplementedError("Currently only coattn mode is fully supported for classification.")
        
        if stop: 
            break

    # 모델 최고 성능 지점 로드
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    
    # 수정: summary_classification_coattn에서 딕셔너리, Acc, AUC 반환
    results_val_dict, val_acc, val_auc = summary_classification_coattn(model, val_loader, args.n_classes)
    
    print('Val Accuracy: {:.4f}, Val AUC: {:.4f}'.format(val_acc, val_auc))
    if writer:
        writer.close()
        
    return results_val_dict, val_acc, val_auc