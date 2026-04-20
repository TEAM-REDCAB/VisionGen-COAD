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
        csv_path='dataset_csv/ccrcc_clean.csv', mode='omic', apply_sig=False,
        shuffle=False, seed=7, print_info=True, 
        patient_strat=False, label_col='msi_status', label_dict={'MSS': 0, 'MSI-H': 1}):
        r"""
        Generic_WSI_Classification_Dataset 
        (생존 분석용에서 분류용으로 리팩토링됨)
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        # 1. 데이터 불러오기 및 기본 처리
        slide_data = pd.read_csv(csv_path, low_memory=False)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        self.label_col = label_col
        assert self.label_col in slide_data.columns, f"{self.label_col} 컬럼이 CSV 파일에 없습니다."

        # BRCA 데이터셋의 경우 IDC만 필터링 (기존 로직 유지)
        if "IDC" in slide_data['oncotree_code'].values:
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # 2. 분류용 라벨 매핑 (문자열 -> 0, 1)
        # 결측치가 있는 행 제거
        slide_data = slide_data.dropna(subset=[label_col])
        slide_data['label'] = slide_data[label_col].map(label_dict)
        
        # 매핑 안 된 라벨이 있다면 에러 처리 혹은 제거
        slide_data = slide_data.dropna(subset=['label'])
        slide_data['label'] = slide_data['label'].astype(int)
        
        self.label_dict = label_dict
        self.num_classes = len(set(label_dict.values()))

        # 3. 환자 단위 딕셔너리 생성
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        patient_dict = {}
        slide_data_indexed = slide_data.set_index('case_id')
        
        for patient in patients_df['case_id']:
            slide_ids = slide_data_indexed.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict
    
        # 슬라이드 데이터 정리
        slide_data = patients_df.reset_index(drop=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])
        
        # disc_label(연속형 변수 구간 라벨) 삭제, 단순 label만 유지
        self.patient_data = {'case_id': slide_data['case_id'].values, 'label': slide_data['label'].values}

        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12] # 메타데이터 컬럼 조정 필요시 수정
        self.mode = mode
        self.cls_ids_prep()

        # 4. Signatures (기존 로직 유지)
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            label = self.slide_data['label'][locations[0]]
            patient_labels.append(label)
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: \n", self.slide_data['label'].value_counts(sort=False))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split


class Generic_MIL_Classification_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Classification_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False # 기본값은 False지만, main.py에서 load_from_h5(True)를 호출하면 True로 바뀜

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['label'][idx] # 이진 분류 라벨
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if self.data_dir:
            # === WSI 피처 로딩을 pt/h5 모두 지원하도록 통합 (중복 코드 제거) ===
            def load_wsi_features(slide_ids, base_dir):
                path_features = []
                for slide_id in slide_ids:
                    slide_id = slide_id.rstrip('.svs')
                    if not self.use_h5:
                        wsi_path = os.path.join(base_dir, 'pt_files', f'{slide_id}.pt')
                        wsi_bag = torch.load(wsi_path)
                    else:
                        wsi_path = os.path.join(base_dir, 'h5_files', f'{slide_id}.h5')
                        with h5py.File(wsi_path, 'r') as hdf5_file:
                            # h5 파일 내 'features' 키에서 데이터를 읽어와 텐서로 변환
                            wsi_bag = torch.from_numpy(hdf5_file['features'][:])
                    path_features.append(wsi_bag)
                return torch.cat(path_features, dim=0) if path_features else torch.zeros((1,1))
            # ====================================================================

            if self.mode == 'path':
                path_features = load_wsi_features(slide_ids, data_dir)
                return (path_features, torch.zeros((1,1)), label)

            elif self.mode == 'omic':
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (torch.zeros((1,1)), genomic_features, label)

            elif self.mode == 'pathomic':
                path_features = load_wsi_features(slide_ids, data_dir)
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (path_features, genomic_features, label)

            elif self.mode == 'coattn':
                path_features = load_wsi_features(slide_ids, data_dir)
                
                omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                
                # 분류용 타겟(label)만 반환
                return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label)

            else:
                raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
        else:
            return slide_ids, label


class Generic_Split(Generic_MIL_Classification_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
            self.fname2ids = pickle.load(handle)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                # ✅ DNA 피처의 접미사에 맞게 수정합니다.
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv']]) 
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed