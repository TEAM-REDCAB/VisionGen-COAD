# make_splits.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

CSV_PATH  = './data/tcga_coad_all_clean.csv'
SPLIT_DIR = './splits/5foldcv/tcga_coad_100'
os.makedirs(SPLIT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
# 환자 단위로 split (슬라이드 단위 아님)
patients = df.drop_duplicates('case_id')[['case_id', 'msi_status']]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for fold, (train_idx, val_idx) in enumerate(
        skf.split(patients['case_id'], patients['msi_status'])):
    train_ids = patients.iloc[train_idx]['case_id'].values
    val_ids   = patients.iloc[val_idx]['case_id'].values
    split_df  = pd.DataFrame({
        'train': pd.Series(train_ids),
        'val':   pd.Series(val_ids)
    })
    split_df.to_csv(os.path.join(SPLIT_DIR, f'splits_{fold}.csv'), index=False)
    print(f"fold {fold} — train: {len(train_ids)}, val: {len(val_ids)}")

print("✅ split CSVs 생성 완료")