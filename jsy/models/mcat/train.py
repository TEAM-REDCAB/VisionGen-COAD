import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.mcat_multimodal_dataset import MSI_Multimodal_Dataset, get_label_path
from modules.mcat_model import MCAT_Binary, BinaryFocalLoss
from modules.mcat_train_binary import train_binary, validate_binary
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1. 하이퍼파라미터 설정
    max_epochs = 20
    batch_size = 1
    lr = 1e-4
    gc_steps = 16 # VRAM 12GB 맞춤형 Gradient Accumulation
    results_dir = "./results_msi"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 전체 데이터셋 로드 (이전 단계에서 만든 MSI_Multimodal_Dataset)
    common_patients = "./data/common_patients.txt" # 경로 수정 필요
    feats_path = "/home/team/projects/team_REDCAB/team_project/data/gigapath_processed/20.0x_256px_0px_overlap/features_gigapath"        # 경로 수정 필요
    npy_path = "./data/genomic_input_matrix.npy"
    pkl_path = "./data/genomic_encoding_states.pkl"
    
    labeled_csv_path = get_label_path(common_patients)
    fold_results = []

    # 4. 5-Fold 교차 검증 루프 (새로운 데이터셋 구조 적용)
    for fold_idx in range(5):
        fold_col = f'fold_{fold_idx}'
        print(f"\n{'='*40}")
        print(f"========== Fold {fold_idx} 학습 시작 ==========")
        print(f"{'='*40}")
        
        # Train 및 Val 데이터셋 선언 (이미 나누어진 fold_col 기준)
        train_dataset = MSI_Multimodal_Dataset(
            split='train', 
            fold_col=fold_col, 
            csv_path=labeled_csv_path, # 새로 생성된 CSV 사용
            feats_path=feats_path, 
            npy_path=npy_path, 
            pkl_path=pkl_path
        )
        
        val_dataset = MSI_Multimodal_Dataset(
            split='val', 
            fold_col=fold_col, 
            csv_path=labeled_csv_path, 
            feats_path=feats_path, 
            npy_path=npy_path, 
            pkl_path=pkl_path
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 및 옵티마이저 초기화 (폴드마다 가중치 초기화)
        model = MCAT_Binary().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        # loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        
        best_val_auprc = 0.0
        best_val_auroc = 0.0

        # --- 🚨 얼리 스토핑 관련 변수 추가 ---
        patience = 10  # 최고 성능을 갱신하지 못하고 5에폭을 버티면 종료
        early_stop_counter = 0
        best_thresh = 0
        
        for epoch in range(1, max_epochs + 1):
            train_binary(epoch, model, train_loader, optimizer, loss_fn, gc=gc_steps)
            val_auroc, val_auprc,val_thresh = validate_binary(epoch, model, val_loader, loss_fn)
            
            # 기준을 AUROC로 변경하여 최고의 분류 성능을 가진 모델 가중치를 저장
            if val_auroc > best_val_auroc:
                best_val_auprc = val_auprc
                best_val_auroc = val_auroc
                best_thresh = val_thresh
                checkpoint_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pt")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'best_thresh': val_thresh,  # 👈 여기서 최적 임계값 저장
                    'auprc': val_auprc,
                    'auroc':val_auroc
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"🔥 Fold {fold_idx} 최고 성능 갱신! 모델 저장됨 (AUROC: {val_auroc:.4f}), (AUPRC: {val_auprc:.4f}), Best Threshold: {val_thresh:.4f}")
                # 카운터 초기화
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"{patience} epoch 동안 성능 개선이 없어 Fold {fold_idx} 학습 조기 종료")
                break
                
        fold_results.append(best_val_auroc)
        print(f"Fold {fold_idx} 종료. AUROC: {best_val_auroc:.4f}, AUPRC: {best_val_auprc:.4f}, 최적 Threshold: {best_thresh:.4f}")

    # 최종 결과 출력
    print("\n================ 최종 결과 ================")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1} Best AUROC: {acc:.4f}")
    print(f"5-Fold 평균 AUROC: {np.mean(fold_results):.4f}")

if __name__ == '__main__':
    main()