import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_PATH = '2060'  # gigapath, uni_v2, cptac...
SEED = 1


def get_feats_path(test=False):
    if test:
        return '/home/team/projects/team_REDCAB/team_project/data/gigapath_processed_cptac/20.0x_256px_0px_overlap/features_gigapath'
    if DATA_PATH == 'gigapath':
        return '/home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath'
    elif DATA_PATH == 'uni_v2':
        return '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
    elif DATA_PATH == 'cptac':
        return '/home/team1/data/cptac_processed/20.0x_256px_0px_overlap/features_gigapath'
    elif DATA_PATH == 'cptac_2060':
        return '/home/team/projects/team_REDCAB/team_project/data/gigapath_processed_cptac/20.0x_256px_0px_overlap/features_gigapath'
    else:
        return '/home/team/projects/team_REDCAB/team_project/data/gigapath_processed_tcga/20.0x_256px_0px_overlap/features_gigapath'


def get_coords_path(test=False):
    if test:
        return '/home/team/projects/team_REDCAB/team_project/data/gigapath_processed_cptac/20.0x_256px_0px_overlap/patches'
    if DATA_PATH == 'gigapath':
        return '/home/team1/data/gigapath_processed/20.0x_256px_0px_overlap/patches'
    elif DATA_PATH == 'uni_v2':
        return '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/patches'
    elif DATA_PATH == 'cptac':
        return '/home/team1/data/cptac_processed/20.0x_256px_0px_overlap/patches'
    elif DATA_PATH == 'cptac_2060':
        return '/home/team/projects/team_REDCAB/team_project/data/gigapath_processed_cptac/20.0x_256px_0px_overlap/patches'
    else:
        return '/home/team/projects/team_REDCAB/team_project/data/gigapath_processed_tcga/20.0x_256px_0px_overlap/patches'


def get_label_path(test=False):
    if test:
        label_path = os.path.join('./labels', f'clinical_data_seed_{SEED}_test.csv')
        if os.path.exists(label_path):
            return label_path
        df = pd.read_csv('./labels/cptac_coad_patient_msi.csv')
        df = df.rename(columns={'patient_id': 'patient'})
        df['msi'] = df['msi_status'].map({'MSS': 0, 'MSI-H': 1})
        for i in range(5):
            df[f'fold_{i}'] = 'test'
        os.makedirs('labels', exist_ok=True)
        df.to_csv(label_path, index=False)
        return label_path
    label_path = os.path.join('./labels', f'clinical_data_seed_{SEED}.csv')
    if os.path.exists(label_path):
        return label_path
    df = pd.read_csv('./labels/common_patients.txt', sep='\t')
    df['msi'] = df['type'].map({'MSS': 0, 'MSIMUT': 1})

    for i in range(5):
        df[f'fold_{i}'] = 'none'
    if SEED != 1:
        df_train_val, df_test = train_test_split(
            df,
            test_size=0.2,
            stratify=df['msi'],
            random_state=SEED
        )
    else:
        df_train_val = df

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['msi'])):
        actual_train_idx = df_train_val.iloc[train_idx].index
        actual_val_idx = df_train_val.iloc[val_idx].index

        df.loc[actual_train_idx, f'fold_{fold_idx}'] = 'train'
        df.loc[actual_val_idx, f'fold_{fold_idx}'] = 'val'
        if SEED != 1:
            df.loc[df_test.index, f'fold_{fold_idx}'] = 'test'

    os.makedirs('labels', exist_ok=True)
    df.to_csv(label_path, index=False)
    return label_path


def get_results_path():
    results_path = os.path.join('./results', f'seed_{SEED}')
    os.makedirs(results_path, exist_ok=True)
    return results_path


def get_teacher_knowledge_path():
    return "./models/mcat/teacher_knowledge"
