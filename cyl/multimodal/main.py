# workflow image에서 4번에 해당하는 코드

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Pathomic_Classification_Dataset
from model import MCAT_Single_Branch_Model

# =========================================================================
# 검은색 화살표 (파이프라인 통제 사령부 - REDCAB_MCAT 버전)
# =========================================================================
def main():
    print("=== [NPY 호환 모드] 진짜 MCAT (맞춤 해독기 탑재) 학습 스크립트 시작 ===\n")
    
    # [경로 설정]
    clin_csv = './dataset_csv/ccrcc_clean.csv' 
    mut_csv = './preprocessed_mutation_data.csv' 
    npy_file = './genomic_input_matrix.npy'      
    pkl_file = './genomic_encoding_states.pkl'   # 팀원님이 만든 단어장(규칙)
    wsi_dir = './data_features'                  
    
    # 0. 단어 사전(Vocabulary) 로드하기
    print("-> 0/3 사전 데이터(.pkl) 확보 중...")
    try:
        with open(pkl_file, 'rb') as f:
            encoding_states = pickle.load(f)
            
        # 사전에 적힌 단어 총량 계산
        vocab_sizes = {
            'var': len(encoding_states['var_vocab']),
            'vc': len(encoding_states['vc_vocab']),
            'func': len(encoding_states['func_vocab'])
        }
        print(f"[성공] 사전 로딩 성공! (변이종류: {vocab_sizes['var']}개 등)")
    except FileNotFoundError:
        print("[경고] pkl 파일을 찾을 수 없습니다. (에러 방지용 임시 사전 크기 할당)")
        # 데이터가 없을 때 돌아가는지 테스트하기 위한 모의 사이즈
        vocab_sizes = {'var': 1500, 'vc': 25, 'func': 100}

    # 1. 데이터 로더 섭외 (2번 상자)
    print("-> 1/3 데이터로더(2번 상자) 준비 중...")
    # dataset = Pathomic_Classification_Dataset(
    #     clin_csv_path=clin_csv, mut_csv_path=mut_csv, npy_path=npy_file, data_dir=wsi_dir
    # )
    # MCAT은 환자 1명씩 처리해야 하므로 batch_size=1
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. 모델 섭외 (3번 상자)
    print("-> 2/3 REDCAB_MCAT(3번 상자) 세팅 중... (단어 해독기 연동 완료!)")
    model = MCAT_Single_Branch_Model(vocab_sizes=vocab_sizes, path_dim=1024, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. 전송 시작 
    print("-> 3/3 학습 파이프라인 연결 완료 (미가동 상태)\n")
    model.train()
    
    # for batch in dataloader:
    #     path_features, genomic_features, label = batch
    #     path_features = path_features.squeeze(0)
    #     
    #     logits, Y_hat, attn_scores = model(path_features, genomic_features)
    #     loss = loss_fn(logits, label)
    #     
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    print("[성공] 우리 데이터 전용 번역기를 장착한 REDCAB_MCAT 코드가 준비되었습니다!")

if __name__ == '__main__':
    main()
