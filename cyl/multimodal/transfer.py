# workflow image에서 ‘GPU 메모리 전달’에 해당하는 코드

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 2번, 3번 상자 불러오기
from dataset_1branch import Pathomic_Classification_Dataset
from model_1branch import Pathomic_Single_Array_Model

# =========================================================================
# 검은색 화살표 (GPU 메모리로 전달)
# =========================================================================
def main():
    print("=== [NPY 호환 모드] 메인 학습 스크립트 시작 ===\n")
    
    # [경로 설정] 환경에 맞춰야 하는 값
    clin_csv = './dataset_csv/ccrcc_clean.csv' # 기존의 WSI, 라벨이 있는 파일
    mut_csv = './preprocessed_mutation_data.csv' # 팀원이 만든 중간 매칭 파일
    npy_file = './genomic_input_matrix.npy'      # 팀원이 최종 산출한 큐브 파일
    wsi_dir = './data_features'                  # 이미지 특징 파일 위치
    
    # 1. 데이터 로더 섭외 (2번 상자)
    print("-> 1/3 데이터로더(2번 상자) 준비 중... (.npy 통합 완료)")
    dataset = Pathomic_Classification_Dataset(
         clin_csv_path=clin_csv, mut_csv_path=mut_csv, npy_path=npy_file, data_dir=wsi_dir
     )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. 모델 섭외 (3번 상자)
    print("-> 2/3 딥러닝 모델(3번 상자) 세팅 중... (입력 통로 9개짜리 딥-세트(DeepSets) 적용 완료)")
    # seq_dim은 팀원분이 설계하신 9로 픽스합니다.
    model = Pathomic_Single_Array_Model(path_dim=1024, seq_dim=9, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. 무한반복 학습 루프
    print("-> 3/3 학습 파이프라인 연결 완료 (미가동 상태)")
    model.train()
    
    # for batch in dataloader:
    #     path_features, genomic_features, label = batch
    #     # genomic_features의 모양이 여기서 (Batch, 1425, 9) 형태로 돌아다니게 됩니다.
    #     
    #     logits, _, _ = model(path_features, genomic_features)
    #     loss = loss_fn(logits, label)
    #     
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    print("\n[성공] 팀원의 1425x9 데이터를 흡수하는 세팅이 완료되었습니다!")

if __name__ == '__main__':
    main()
