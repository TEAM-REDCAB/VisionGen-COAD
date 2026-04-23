from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth

# =========================================================================

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
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
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--
