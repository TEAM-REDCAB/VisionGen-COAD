import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.mcat_multimodal_dataset import MSI_Multimodal_Dataset, get_label_path, get_feats_path
from modules.mcat_model import MCAT_Binary, BinaryFocalLoss
from modules.mcat_train_binary import train_binary
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1. 하이퍼파라미터 설정
    max_epochs = 5
    batch_size = 1
    lr = 1e-4
    gc_steps = 16 # VRAM 12GB 맞춤형 Gradient Accumulation
    results_dir = "./results_msi"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 전체 데이터셋 로드 (이전 단계에서 만든 MSI_Multimodal_Dataset)
    common_patients = "./data/common_patients.txt" # 경로 수정 필요
    feats_path = get_feats_path()
    npy_path = "./data/genomic_input_matrix.npy"
    pkl_path = "./data/genomic_encoding_states.pkl"
    
    labeled_csv_path = get_label_path(common_patients)

    # 4. 5-Fold 교차 검증 루프 (새로운 데이터셋 구조 적용)
    # Train 및 Val 데이터셋 선언 (이미 나누어진 fold_col 기준)
    train_dataset = MSI_Multimodal_Dataset(
        split='all', 
        fold_col='fold_1', 
        csv_path=labeled_csv_path, # 새로 생성된 CSV 사용
        feats_path=feats_path, 
        npy_path=npy_path, 
        pkl_path=pkl_path
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 및 옵티마이저 초기화 (폴드마다 가중치 초기화)
    model = MCAT_Binary().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    
    best_val_auprc = 0.0
    best_val_auroc = 0.0

    # --- 🚨 얼리 스토핑 관련 변수 추가 ---
    best_thresh = 0
    
    for epoch in range(1, max_epochs + 1):
        val_auroc, val_auprc, val_thresh = train_binary(epoch, model, train_loader, optimizer, loss_fn, gc=gc_steps)
        
        # 기준을 AUROC로 변경하여 최고의 분류 성능을 가진 모델 가중치를 저장
        if val_auroc > best_val_auroc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_thresh = val_thresh
            checkpoint_path = os.path.join(results_dir, "best_model_full.pt")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_thresh': val_thresh,  # 👈 여기서 최적 임계값 저장
                'auprc': val_auprc,
                'auroc':val_auroc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"🔥 최고 성능 갱신! 모델 저장됨 (AUROC: {val_auroc:.4f}), (AUPRC: {val_auprc:.4f}), Best Threshold: {val_thresh:.4f}")

if __name__ == '__main__':
    main()