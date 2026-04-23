import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

ENCODER_MODEL='gigapath'    # gigapath, uni_v2
SEED = 42

def get_feats_path():
    return '/home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath'

def get_coords_path():
    coords_path = '/home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/patches_gigapath'
    return coords_path

def get_results_path():
    results_path = os.path.join('./results', f'seed_{SEED}')
    os.makedirs(results_path, exist_ok=True)
    return results_path
    
def get_label_path():
    df = pd.read_csv('./labels/common_patients.txt', sep='\t')
    df['msi'] = df['type'].map({'MSS':0, 'MSIMUT':1})

    df_train_val, df_test = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['msi'], 
        random_state=SEED  
    )

    for i in range(5):
        df[f'fold_{i}'] = 'none'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['msi'])):
        actual_train_idx = df_train_val.iloc[train_idx].index
        actual_val_idx = df_train_val.iloc[val_idx].index
        
        df.loc[actual_train_idx, f'fold_{fold_idx}'] = 'train'
        df.loc[actual_val_idx, f'fold_{fold_idx}'] = 'val'
        df.loc[df_test.index, f'fold_{fold_idx}'] = 'test'

    os.makedirs('labels', exist_ok=True)
    label_path = os.path.join('labels', f'clinical_data_seed_{SEED}.csv')
    df.to_csv(label_path, index=False)
    return label_path
