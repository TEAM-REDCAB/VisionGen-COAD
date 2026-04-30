import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# 기존에 작성한 모듈 임포트
from modules.mcat_multimodal_dataset import MSI_Multimodal_Dataset, get_label_path, get_feats_path
from modules.mcat_model import MCAT_Binary

def evaluate_test_set(result_path, csv_path, feats_path, npy_path, pkl_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====> 실전 테스트 시작 <====")
    # 1. 완전한 Unseen 테스트 데이터셋 로드
    # 생성된 CSV 파일에서 fold_0 컬럼에 'test'라고 마킹된 20%의 데이터를 불러옵니다.
    test_dataset = MSI_Multimodal_Dataset(
        split='test', 
        fold_col='fold_0', # 어느 폴드 컬럼이든 test 셋의 인덱스는 동일합니다.
        csv_path=csv_path,
        feats_path=feats_path,
        npy_path=npy_path,
        pkl_path=pkl_path
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"테스트 환자 수: {len(test_dataset)}명\n")

    model = MCAT_Binary().to(device)
    model.eval() # 평가 모드 전환 (Dropout 등 비활성화)
    # 5폴드의 모델을 전부 테스트하기 위해 반복
    for fold_idx in range(5):
    # 2. 모델 초기화 및 학습된 가중치(Best Model) 장착
        model_path = os.path.join(result_path, f"best_model_fold{fold_idx}.pt")
        print(f"로드할 모델 가중치: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_thresh = checkpoint['best_thresh']

        all_labels = []
        all_probs = []
        all_preds = []

        # 3. 실전 인퍼런스 루프
        pbar = tqdm(test_loader, desc="Testing", dynamic_ncols=True)
        
        with torch.no_grad(): # 테스트 단계이므로 역전파 금지
            for data_wsi, data_omic, label in pbar:
                data_wsi = data_wsi.to(device)
                data_omic = data_omic.to(device)
                label = label.type(torch.FloatTensor).to(device)

                # (정상적인 멀티모달 테스트 시 아래 라인 사용)
                logits, *_ = model(data_wsi, data_omic)

                # 차원 정리
                logits = logits.squeeze(dim=-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)

                # 시그모이드 통과하여 0~1 사이 확률값 도출
                probs = torch.sigmoid(logits)
                preds = (probs > best_thresh).float() # 0.5 초과면 MSI-H(1), 아니면 MSS(0)

                all_labels.extend(label.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 4. 의료 AI 표준 평가 지표 계산
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.5
            
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        # 5. 최종 리포트 출력
        print("\n" + "="*40)
        print("========== 🏆 최종 테스트 결과 ==========")
        print("="*40)
        print(f"AUROC     : {auroc:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"Threshold : {best_thresh:.4f} ")
        print("\n[Confusion Matrix]")
        print(" TN(MSS맞춤)  FP(MSI로오해)")
        print(" FN(MSS로오해) TP(MSI맞춤)")
        print(cm)
        
        report = classification_report(all_labels, all_preds, target_names=["MSS (0)", "MSI-H (1)"], zero_division=0)
        print("\n[Classification Report]")
        print(report)

if __name__ == '__main__':
    # 테스트에 필요한 경로 설정
    PATIENTS_LABEL = "./data/common_patients.txt"
    CSV_PATH = get_label_path(PATIENTS_LABEL)
    FEATS_PATH = get_feats_path()
    NPY_PATH = "./data/genomic_input_matrix.npy"
    PKL_PATH = "./data/genomic_encoding_states.pkl"
    RESULT_PATH = "./results_msi"
    evaluate_test_set(RESULT_PATH, CSV_PATH, FEATS_PATH, NPY_PATH, PKL_PATH)