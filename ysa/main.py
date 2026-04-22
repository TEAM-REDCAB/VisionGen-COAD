from __future__ import print_function

import argparse
import os
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
# 수정: 분류용 데이터셋 Import
from datasets.dataset_classification import Generic_WSI_Classification_Dataset, Generic_MIL_Classification_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train # (주의: core_utils의 train 함수도 acc, auc를 리턴하도록 맞춰져 있어야 합니다)
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def main(args):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
       os.mkdir(args.results_dir)

    if args.k_start == -1:
       start = 0
    else:
       start = args.k_start
    if args.k_end == -1:
       end = args.k
    else:
       end = args.k_end

    # 수정: cindex 대신 정확도(Acc)와 AUC를 추적할 리스트 생성
    latest_val_acc = []
    latest_val_auc = []
    folds = np.arange(start, end)

    ### Start 5-Fold CV Evaluation.
    for i in folds:
       start_time = timer()
       seed_torch(args.seed)
       results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
       if os.path.isfile(results_pkl_path):
          print("Skipping Split %d" % i)
          continue

       ### Gets the Train + Val Dataset Loader.
       train_dataset, val_dataset = dataset.return_splits(from_id=False, 
             csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
       
       print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
       datasets = (train_dataset, val_dataset)
       
       ### Specify the input dimension size if using genomic features.
       if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
          args.omic_input_dim = train_dataset.genomic_features.shape[1]
          print("Genomic Dimension", args.omic_input_dim)
       elif 'coattn' in args.mode:
          args.omic_sizes = train_dataset.omic_sizes
          print('Genomic Dimensions', args.omic_sizes)
       else:
          args.omic_input_dim = 0

       ### Run Train-Val on Classification Task.
       if args.task_type == 'classification':
          # 수정: core_utils의 train 함수가 (환자별 결과, Acc, AUC)를 반환한다고 가정
          val_latest, acc_latest, auc_latest = train(datasets, i, args) 
          latest_val_acc.append(acc_latest)
          latest_val_auc.append(auc_latest)

       ### Write Results for Each Split to PKL
       save_pkl(results_pkl_path, val_latest)
       end_time = timer()
       print('Fold %d Time: %f seconds' % (i, end_time - start_time))

    ### Finish 5-Fold CV Evaluation.
    if args.task_type == 'classification':
       # 수정: CSV에 cindex 대신 acc와 auc 기록
       results_latest_df = pd.DataFrame({'folds': folds, 'val_acc': latest_val_acc, 'val_auc': latest_val_auc})

    if len(folds) != args.k:
       save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
       save_name = 'summary.csv'

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))


### Training settings
parser = argparse.ArgumentParser(description='Configurations for Classification Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--dataset_path',    type=str, default='./data', help='Directory for csvs')
parser.add_argument('--seed',          type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k',             type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',        type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',         type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_blca_100', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',       action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['snn', 'deepset', 'amil', 'mi_fcn', 'mcat'], default='mcat', help='Type of model (Default: mcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='concat', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',      action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

### Optimizer Parameters + Classification Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',              type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['ce', 'svm'], default='ce', help='slide-level classification loss function (default: ce)') # 수정: CE loss를 기본으로
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg',             type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')
parser.add_argument( '--genomic_dir',    type=str, default='path/to/preprocessing', help='genomic_input_matrix.npy + genomic_encoding_states.pkl 폴더') # genomic_input_matrix.npy가 있는 폴더 
parser.add_argument('--testing', action='store_true', default=False, help='Enable testing mode')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
# 수정: task 이름을 survival에서 classification으로 변경
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_classification'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1536
settings = {'num_splits': args.k, 
          'k_start': args.k_start,
          'k_end': args.k_end,
          'task': args.task,
          'max_epochs': args.max_epochs, 
          'results_dir': args.results_dir, 
          'lr': args.lr,
          'experiment': args.exp_code,
          'reg': args.reg,
          'label_frac': args.label_frac,
          'bag_loss': args.bag_loss,
          'bag_weight': args.bag_weight,
          'seed': args.seed,
          'model_type': args.model_type,
          'model_size_wsi': args.model_size_wsi,
          'model_size_omic': args.model_size_omic,
          "use_drop_out": args.drop_out,
          'weighted_sample': args.weighted_sample,
          'gc': args.gc,
          'opt': args.opt}
print('\nLoad Dataset')

# 수정: 생존 분석용 코드 블록을 이진 분류용으로 전면 교체
if 'classification' in args.task:
    args.n_classes = 2 # MSI, MSS 2개 클래스
    study = '_'.join(args.task.split('_')[:2])
    if study == 'tcga_kirc' or study == 'tcga_kirp':
       combined_study = 'tcga_kidney'
    elif study == 'tcga_luad' or study == 'tcga_lusc':
       combined_study = 'tcga_lung'
    else:
       combined_study = study
    
    study_dir = '%s_20x_features' % combined_study
    
    dataset = Generic_MIL_Classification_Dataset(
        csv_path='%s/%s_all_clean.csv' % (args.dataset_path, combined_study),
        genomic_dir=args.genomic_dir,
        mode=args.mode,
        apply_sig=args.apply_sig,
        data_dir=args.data_root_dir,
        shuffle=False, 
        seed=args.seed, 
        print_info=True,
        patient_strat=False,
        label_col='msi_status', # 이전에 데이터셋에서 설정한 라벨 컬럼
        label_dict={'MSS': 0, 'MSI': 1}
    )
    
    # h5 파일을 사용하신다고 하셨으니 toggle을 True로 설정해줍니다. (.pt 사용시 아래 줄 삭제)
    dataset.load_from_h5(True) 
elif 'coattn' in args.mode:
   args.omic_sizes = dataset.omic_sizes #npy에서 자동 계산된 값 사용
   print('Genomic omic_sizes:', args.omic_sizes)
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Classification_Dataset):
    args.task_type = 'classification'
else:
    raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

### Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start_timer = timer()
    results = main(args)
    end_timer = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end_timer - start_timer))