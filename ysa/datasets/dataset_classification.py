"""
datasets/dataset_classification.py
====================================
TCGA-COAD MSI/MSS 이진 분류용 데이터셋
- WSI 피처: H5 파일 (CLAM 추출, key='features')
- Genomic 피처: prepare_genomic_features.py 로 생성한 CSV
  컬럼 규칙: v{Variant_ID}_g{0~5}  → 자동으로 6개 omic 그룹 감지
"""

from __future__ import print_function, division
import os
import pickle

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
        csv_path='dataset_csv/tcga_coad_all_clean.csv',
        mode='coattn',
        apply_sig=False,
        shuffle=False,
        seed=7,
        print_info=True,
        patient_strat=False,
        label_col='msi_status',
        label_dict={'MSS': 0, 'MSI-H': 1},
    ):
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None

        # ── 1. CSV 로드 ──────────────────────────────────────────────
        slide_data = pd.read_csv(csv_path, low_memory=False)

        if 'case_id' not in slide_data.columns:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        self.label_col = label_col
        assert self.label_col in slide_data.columns, \
            f"'{self.label_col}' 컬럼이 CSV에 없습니다. 컬럼 목록: {slide_data.columns.tolist()}"

        # ── 2. oncotree_code 필터 (컬럼 없으면 스킵) ─────────────────
        # BUG5 수정: 컬럼 존재 여부 먼저 확인
        if 'oncotree_code' in slide_data.columns and 'IDC' in slide_data['oncotree_code'].values:
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # ── 3. 라벨 매핑 ─────────────────────────────────────────────
        slide_data = slide_data.dropna(subset=[label_col])
        slide_data['label'] = slide_data[label_col].map(label_dict)
        slide_data = slide_data.dropna(subset=['label'])
        slide_data['label'] = slide_data['label'].astype(int)

        self.label_dict = label_dict
        self.num_classes = len(set(label_dict.values()))

        # ── 4. 환자 단위 딕셔너리 ────────────────────────────────────
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        slide_data_indexed = slide_data.set_index('case_id')
        patient_dict = {}
        for patient in patients_df['case_id']:
            slide_ids = slide_data_indexed.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array([slide_ids])
            else:
                slide_ids = slide_ids.values
            patient_dict[patient] = slide_ids
        self.patient_dict = patient_dict

        slide_data = patients_df.reset_index(drop=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        self.patient_data = {
            'case_id': slide_data['case_id'].values,
            'label':   slide_data['label'].values,
        }
        self.slide_data = slide_data

        # 메타 컬럼: case_id, slide_id, msi_status, label + oncotree_code 등
        meta_cols = ['case_id', 'slide_id', label_col, 'label']
        if 'oncotree_code' in slide_data.columns:
            meta_cols.append('oncotree_code')
        self.metadata = [c for c in meta_cols if c in slide_data.columns]

        self.mode = mode
        self.cls_ids_prep()

        # ── 5. Signatures (apply_sig=True 시) ────────────────────────
        self.apply_sig = apply_sig
        self.signatures = (
            pd.read_csv('./dataset_csv_sig/signatures.csv') if apply_sig else None
        )

        if print_info:
            self.summarize()

    # ──────────────────────────────────────────────────────────────────
    def cls_ids_prep(self):
        self.patient_cls_ids = [
            np.where(self.patient_data['label'] == i)[0]
            for i in range(self.num_classes)
        ]
        self.slide_cls_ids = [
            np.where(self.slide_data['label'] == i)[0]
            for i in range(self.num_classes)
        ]

    def getlabel(self, idx):
        """utils.py의 make_weights_for_balanced_classes_split에서 사용"""
        return self.slide_data['label'].iloc[idx]

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        return len(self.slide_data)

    def summarize(self):
        print(f"label column: {self.label_col}")
        print(f"label dictionary: {self.label_dict}")
        print(f"number of classes: {self.num_classes}")
        print("slide-level counts:\n", self.slide_data['label'].value_counts(sort=False))

    def get_split_from_df(self, all_splits, split_key='train', scaler=None):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) == 0:
            return None
        mask = self.slide_data['slide_id'].isin(split.tolist())
        df_slice = self.slide_data[mask].reset_index(drop=True)
        return Generic_Split(
            df_slice,
            metadata=self.metadata,
            mode=self.mode,
            signatures=self.signatures,
            data_dir=self.data_dir,
            label_col=self.label_col,
            patient_dict=self.patient_dict,
            num_classes=self.num_classes,
        )

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            raise NotImplementedError
        assert csv_path
        all_splits = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits, split_key='train')
        val_split   = self.get_split_from_df(all_splits, split_key='val')

        print("****** Normalizing Genomic Features ******")
        scalers = train_split.get_scaler()
        train_split.apply_scaler(scalers)
        val_split.apply_scaler(scalers)

        return train_split, val_split


class Generic_MIL_Classification_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, mode='coattn', **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id  = self.slide_data['case_id'].iloc[idx]
        label    = self.slide_data['label'].iloc[idx]
        slide_ids = self.patient_dict[case_id]

        data_dir = (
            self.data_dir[self.slide_data['oncotree_code'].iloc[idx]]
            if isinstance(self.data_dir, dict)
            else self.data_dir
        )

        if not self.data_dir:
            return slide_ids, label

        # ── WSI 피처 로드 ────────────────────────────────────────────
        def load_wsi(slide_ids, base_dir):
            bags = []
            for sid in slide_ids:
                sid = sid.rstrip('.svs')
                if not self.use_h5:
                    path = os.path.join(base_dir, 'pt_files', f'{sid}.pt')
                    bags.append(torch.load(path))
                else:
                    path = os.path.join(base_dir, 'h5_files', f'{sid}.h5')
                    with h5py.File(path, 'r') as f:
                        bags.append(torch.from_numpy(f['features'][:]))
            return torch.cat(bags, dim=0) if bags else torch.zeros((1, 1024))

        if self.mode == 'path':
            path_features = load_wsi(slide_ids, data_dir)
            return (path_features, torch.zeros((1, 1)), label)

        elif self.mode == 'omic':
            genomic = torch.tensor(self.genomic_features.iloc[idx].values, dtype=torch.float32)
            return (torch.zeros((1, 1)), genomic, label)

        elif self.mode == 'pathomic':
            path_features = load_wsi(slide_ids, data_dir)
            genomic = torch.tensor(self.genomic_features.iloc[idx].values, dtype=torch.float32)
            return (path_features, genomic, label)

        elif self.mode == 'coattn':
            path_features = load_wsi(slide_ids, data_dir)
            omics = [
                torch.tensor(
                    self.genomic_features[cols].iloc[idx].values,
                    dtype=torch.float32
                )
                for cols in self.omic_names
            ]
            return (path_features, *omics, label)  # 8-tuple

        else:
            raise NotImplementedError(f"mode '{self.mode}' not implemented.")


class Generic_Split(Generic_MIL_Classification_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None,
                 data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5        = False
        self.slide_data    = slide_data
        self.metadata      = metadata
        self.mode          = mode
        self.data_dir      = data_dir
        self.num_classes   = num_classes
        self.label_col     = label_col
        self.patient_dict  = patient_dict
        self.signatures    = signatures

        self.slide_cls_ids = [
            np.where(self.slide_data['label'] == i)[0]
            for i in range(self.num_classes)
        ]

        # ── Genomic 피처 컬럼 추출 ───────────────────────────────────
        self.genomic_features = self.slide_data.drop(
            columns=[c for c in metadata if c in self.slide_data.columns],
            errors='ignore',
        )

        # ── BUG2 수정: cluster 모드일 때만 pkl 로드 ──────────────────
        if self.mode == 'cluster':
            pkl_path = os.path.join(data_dir, 'fast_cluster_ids.pkl')
            with open(pkl_path, 'rb') as f:
                self.fname2ids = pickle.load(f)
        else:
            self.fname2ids = {}

        # ── omic 그룹 분할 ───────────────────────────────────────────
        # 우선순위 1: signatures.csv로 그룹 지정 (apply_sig=True)
        # 우선순위 2: 컬럼명의 _g{i} suffix로 자동 감지
        # 우선순위 3: 균등 6분할 fallback
        self.omic_names = self._build_omic_names()
        self.omic_sizes = [len(g) for g in self.omic_names]

        print(f"Genomic feature shape: {self.genomic_features.shape}")
        print(f"omic_sizes: {self.omic_sizes}")

    # ──────────────────────────────────────────────────────────────────
    def _build_omic_names(self):
        """omic 6그룹 컬럼 목록 생성"""
        all_cols = list(self.genomic_features.columns)

        # ── 방법 1: signatures.csv 기반 ─────────────────────────────
        if self.signatures is not None:
            def series_intersection(s1, s2):
                return pd.Series(list(set(s1) & set(s2)))

            omic_names = []
            for col in self.signatures.columns:
                genes = self.signatures[col].dropna().unique()
                # _mut, _cnv suffix 모두 탐색
                candidates = np.concatenate([genes + sfx for sfx in ['_mut', '_cnv']])
                matched = sorted(series_intersection(candidates, pd.Series(all_cols)).tolist())
                omic_names.append(matched)
            return omic_names

        # ── 방법 2: v{id}_g{i} 컬럼명 suffix 자동 감지 ─────────────
        suffix_groups = [
            [c for c in all_cols if c.endswith(f'_g{i}')]
            for i in range(6)
        ]
        if any(len(g) > 0 for g in suffix_groups):
            # 빈 그룹 경고
            for i, g in enumerate(suffix_groups):
                if len(g) == 0:
                    print(f"  ⚠️  omic{i} 그룹에 피처가 없습니다. Func_IDs 분포 확인 필요.")
            return suffix_groups

        # ── 방법 3: 균등 6분할 fallback ─────────────────────────────
        print("  ℹ️  omic 그룹 자동 감지 실패 → 균등 6분할 fallback 사용")
        n = len(all_cols)
        chunk = max(1, n // 6)
        return [all_cols[i * chunk:(i + 1) * chunk] for i in range(6)]

    # ──────────────────────────────────────────────────────────────────
    def getlabel(self, idx):
        return self.slide_data['label'].iloc[idx]

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        scaler = StandardScaler().fit(self.genomic_features)
        return (scaler,)

    def apply_scaler(self, scalers):
        transformed = pd.DataFrame(
            scalers[0].transform(self.genomic_features),
            columns=self.genomic_features.columns,
        )
        self.genomic_features = transformed