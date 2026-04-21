from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *



##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(MaxNet, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self, **kwargs):
            x = kwargs['x_omic']
            features = self.fc_omic(x)

            logits = self.classifier(features) # [Batch, 2] 형태의 결과값
            Y_prob = F.softmax(logits, dim=1) # 각 클래스(MSS/MSI-H)일 확률
            Y_hat = torch.topk(logits, 1, dim=1)[1] # 가장 높은 확률을 가진 인덱스(0 또는 1)
            
            # MCAT의 리턴 형식과 맞춰주기 위해 (logits, prob, prediction, attention) 순서로 반환
            return logits, Y_prob, Y_hat, None

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)