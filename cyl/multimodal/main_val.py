# workflow image에서 4번에 해당하는 코드

import os
import csv
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
 
from dataset import Pathomic_Classification_Dataset
from model import MCAT_Single_Branch_Model



# =========================================================================
# 검은색 화살표 (파이프라인 통제 사령부 - REDCAB_MCAT 버전)
# =========================================================================
def main():
    print("=== [팀원 NPY 호환 모드] 진짜 MCAT (맞춤 해독기 탑재) 학습 스크립트 시작 ===\n")
    
    # GPU 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> [시스템] 현재 가동 중인 딥러닝 엔진: {device}")
    
    # [로그 저장용 리스트 초기화]
    epoch_summary_log = []   # 최종 결과 (epoch별 요약)
    batch_detail_log = []    # 환자별 순간 오차 로그
    prediction_log = []
    
    # [경로 설정]
    clin_txt = '/home/team1/data/coad_msi_mss/common_patients.txt'                            # 환자들의 이름표와 진짜 정답(MSI인지 MSS인지, 혹은 생존 기간 등)이 나열된 임상 데이터(Clinical) 파일
    mut_csv = '/home/team1/cyl/VisionGen-COAD/cyl/genomicdata/preprocessed_mutation_data.csv' 
    npy_file = '/home/team1/cyl/VisionGen-COAD/cyl/genomicdata/genomic_input_matrix.npy'      
    pkl_file = '/home/team1/cyl/VisionGen-COAD/cyl/genomicdata/genomic_encoding_states.pkl'   # lhj님이 만든 단어장(규칙)
    wsi_dir = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_gigapath'  # jsy님이 만든 WSI 병리 이미지의 .pt 특징 파일들이 잔뜩 모여있는 폴더(디렉토리)
    
  # 0. 단어 사전(Vocabulary) 로드하기
    print("-> 0/3 사전 데이터(.pkl) 확보 중...")
    try:
        with open(pkl_file, 'rb') as f:
            encoding_states = pickle.load(f)
        # pkl 파일에서 func_vocab 로드
        func_vocab = encoding_states['func_vocab']  # {1: 'Cell Differentiation...', 2: 'Cytokines...', 3: 'Oncogenes'}
        var_vocab = encoding_states['var_vocab']    # {1: 'AAK1_c.2278del', ...}
            
        # 사전에 적힌 단어 총량 계산
        vocab_sizes = {
            'var': len(encoding_states['var_vocab']),
            'vc': len(encoding_states['vc_vocab']),
            'func': len(encoding_states['func_vocab'])
        }
        print(f"[성공] 사전 로딩 성공! (변이종류: {vocab_sizes['var']}개 등)")


        # XAI용 역매핑 테이블 생성
        var_vocab = encoding_states['var_vocab']
        vc_vocab = encoding_states['vc_vocab']
        func_vocab = encoding_states['func_vocab']

        var_vocab_inv = {v: k for k, v in var_vocab.items()}  # ID → 이름
        vc_vocab_inv = {v: k for k, v in vc_vocab.items()}
        func_vocab_inv = {v: k for k, v in func_vocab.items()}


    except FileNotFoundError:
        print("[경고] pkl 파일을 찾을 수 없습니다. (에러 방지용 임시 사전 크기 할당)")
        # 데이터가 없을 때 돌아가는지 테스트하기 위한 모의 사이즈
        vocab_sizes = {'var': 1500, 'vc': 25, 'func': 100}
        var_vocab_inv = {}
        vc_vocab_inv = {}
        func_vocab_inv = {}



    # 1. 데이터 로더 섭외 (2번 상자)
    print("-> 1/3 데이터로더(2번 상자) 준비 중...")
    dataset = Pathomic_Classification_Dataset(
         clin_txt_path=clin_txt, mut_csv_path=mut_csv, npy_path=npy_file, data_dir=wsi_dir
     )
     
    # -----[테스트용 소량 절단 코드 추가 부분]--------------------------
    # 전체 수백 명의 환자 중, 앞의 딱 10명만 잘라내서 테스트 모드 돌입!
    # dataset = Subset(dataset, range(10))
    # ----------------------------------------------------------------
     
     
    # 8:2 분할
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset  = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)
    
    print(f"[데이터 분할] 전체: {len(dataset)}명 | 학습: {len(train_dataset)}명 | 테스트: {len(test_dataset)}명")

    
    # 2. 모델 섭외 (3번 상자)
    print("-> 2/3 REDCAB_MCAT(3번 상자) 세팅 중... (단어 해독기 연동 완료!)")
    model = MCAT_Single_Branch_Model(vocab_sizes=vocab_sizes, path_dim=1536, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. 전송 시작       
    print("[성공] 3/3 학습 파이프라인 연결 완료 (가동 준비 끝)\n")
        
    
# =========================================================================
# [실전 채점 훈련 루프]
# =========================================================================
#epochs 수 조절 => 50-100번은 해야함. 현재는 test용

    epochs = 50
    print("멀티모달 구동... 학습 시작!\n")
    
    for epoch in range(epochs):
        # ============================
        # [Train 단계]
        # ============================

        model.train() 
        total_loss = 0.0
        
        y_true_list = []
        y_score_list = []
        y_pred_list = []
        
        for batch_idx, batch in enumerate(train_loader):
            path_features, genomic_features, label, patient_id = batch
            path_features = path_features.squeeze(0).to(device)
            genomic_features = genomic_features.squeeze(0).to(device)
            label = label.to(device)
            
            optimizer.zero_grad() 
            logits, Y_hat, attn_scores = model(path_features, genomic_features) #attn_score: 높을수록 모델이 중요하게 생각했다는 뜻
            
            loss = loss_fn(logits, label)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # --- 채점표에 기록할 항목 ---
            prob      = torch.softmax(logits, dim=1)
            msi_prob  = prob[0, 1].item()
            pred_class = Y_hat.item()        
            
            y_true_list.append(label.item())
            y_score_list.append(msi_prob)
            y_pred_list.append(pred_class)

            batch_detail_log.append({
                'phase': 'train',
                'epoch': epoch + 1,
                'patient_idx': batch_idx + 1,
                'loss': round(loss.item(), 4)
            })

        # 에폭별 Train 평균 지표 계산
        train_loss = total_loss / len(train_loader)
        train_acc  = accuracy_score(y_true_list, y_pred_list)
        try:
            train_auc = roc_auc_score(y_true_list, y_score_list)
        except ValueError:
            train_auc = 0.5

        # ============================
        # [Test 단계]
        # ============================                    
        model.eval()
        test_loss = 0.0
        y_true_test, y_score_test, y_pred_test = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                path_features, genomic_features, label, patient_id = batch
                path_features    = path_features.squeeze(0).to(device)
                genomic_features = genomic_features.squeeze(0).to(device)  # ✅ squeeze 추가!
                label            = label.to(device)
 
                logits, Y_hat, attn_scores = model(path_features, genomic_features)
                loss = loss_fn(logits, label)
                test_loss += loss.item()
 
                prob       = torch.softmax(logits, dim=1)
                msi_prob   = prob[0, 1].item()
                pred_class = Y_hat.item()
 
                y_true_test.append(label.item())
                y_score_test.append(msi_prob)
                y_pred_test.append(pred_class)


                # [MSI/MSS 예측 + XAI 설명 생성]
                patient_id = patient_id[0]  # DataLoader가 리스트로 감싸므로 [0]으로 batch에서 꺼내기
                prediction_label = 'MSIMUT' if pred_class == 1 else 'MSS'
                prob_pct = round(msi_prob * 100 if pred_class == 1 else (1 - msi_prob) * 100, 1)
                true_label = 'MSIMUT' if label.item() == 1 else 'MSS'


                # Co-Attention 기반 XAI 설명
                A_coattn = attn_scores['coattn']  # (1, 1425, N_patches). 어떤 돌연변이가 어떤 WSI 패치를 attend했는지
                omic_importance = A_coattn.squeeze(0).sum(dim=1)  # 각 돌연변이 총 기여도
                top_omic_idx = omic_importance.argmax().item()
    
                top_patch_idx = A_coattn[0, top_omic_idx].argmax().item()
                patch_attention_score = A_coattn[0, top_omic_idx, top_patch_idx].item()            

                # Co-Attention-genomic_features shape: (1425, 9) - squeeze 덕분
                genomic_row = genomic_features[top_omic_idx]  # shape: (9,)
                
                # Co-Attention-각 컬럼 추출
                var_id = int(genomic_row[0].item())
                vc_id = int(genomic_row[1].item())
                func_ids = genomic_row[2:8].long()  # 6개 func_id

                # Co-Attention-역매핑으로 이름 추출
                var_name = var_vocab_inv.get(var_id, f'Unknown_Var#{var_id}')
                vc_name = vc_vocab_inv.get(vc_id, f'Unknown_VC#{vc_id}')
                
                # Co-Attention-활성 기능 추출
                active_funcs = []
                for f_id in func_ids:
                    f_id_val = int(f_id.item())
                    if f_id_val > 0 and f_id_val in func_vocab_inv:
                        active_funcs.append(func_vocab_inv[f_id_val])


                # Co-Attention-최종 설명 생성
                if active_funcs:
                    func_str = ", ".join(active_funcs)
                    reason = f"{var_name}({vc_name}) 돌연변이의 [{func_str}] 기능이 WSI 패치#{top_patch_idx}의 형태 이상을 주목(Attention: {patch_attention_score:.4f}) → {prediction_label} 판별에 기여"
                else:
                    reason = f"{var_name}({vc_name}) 돌연변이가 WSI 패치#{top_patch_idx}의 morphological 패턴 주목(Attention: {patch_attention_score:.4f}) → {prediction_label} 판별에 기여"


                if epoch == epochs - 1:  # 마지막 에폭일 때만 기록!
                    prediction_log.append({
                        'patient': patient_id,
                        'true_label': true_label,
                        'prediction': prediction_label,
                        'probability(%)': prob_pct,
                        'top_variant': var_name,
                        'variant_class': vc_name,
                        'gene_functions': func_str if active_funcs else 'N/A',
                        'top_patch_idx': top_patch_idx,
                        'attention_score': round(patch_attention_score, 4),
                        'reason': reason
                    })

        # 에폭별 Train 평균 지표 계산
        test_loss = test_loss / len(test_loader)
        test_acc  = accuracy_score(y_true_test, y_pred_test)
        try:
            test_auc = roc_auc_score(y_true_test, y_score_test)
        except ValueError:
            test_auc = 0.5



        # ============================
        # [결과 출력 및 로그 기록]
        # ============================
        # [에폭 요약 로그] 매 에폭마다 한 줄씩 저장 (총 50(에포크 총 수)줄 예정)
        epoch_summary_log.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc * 100, 2),
            'train_auc': round(train_auc, 4),
            'test_loss': round(test_loss, 4),
            'test_acc': round(test_acc * 100, 2),
            'test_auc': round(test_auc, 4)
        })
        print(f"[*] Epoch {epoch+1:02d} 완료 | Test Acc: {test_acc*100:.2f}%")
    
    # =========================================================================
    # [3. 최종 저장] 모든 에폭 루프가 끝난 후 실행. csv 파일 저장
    # =========================================================================
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
        

        
    # 요약 정보 저장 (에포크별 평균 성능(50줄))
    summary_path = os.path.join(save_dir, 'epoch_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_summary_log[0].keys())
        writer.writeheader()
        writer.writerows(epoch_summary_log)
    print(f"[저장 완료] Epoch 요약 → {summary_path}")

    # 최종 개별 환자 예측 결과(마지막 에포크 테스트 환자) 저장 (테스트셋 환자 수만큼, 예: 54줄)
    pred_path = os.path.join(save_dir, 'MSI_MSS_prediction.csv')
    with open(pred_path, 'w', newline='', encoding='utf-8-sig') as f:
        # 필드네임을 명시적으로 지정하여 저장
        fieldnames = [
            'patient', 'true_label', 'prediction', 'probability(%)',
            'top_variant', 'variant_class', 'gene_functions',
            'top_patch_idx', 'attention_score', 'reason'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_log)
    print(f"[저장 완료] MSI/MSS 예측 (XAI) → {pred_path}")

 
    # 최종 결과 화면 출력 (마지막 에폭 기준)
    print(f"\n" + "="*40)
    print(f" [최종 채점표] Epoch {epoch+1}/{epochs} Train / Test 결과 요약")
    print(f" Train  →  Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | AUC: {train_auc:.4f}")
    print(f" Test   →  Loss: {test_loss:.4f}  | Acc: {test_acc*100:.2f}%  | AUC: {test_auc:.4f}")
    # print(f" [평균 오차(Loss)] : {avg_loss:.4f}  (낮을수록 좋음)")
    # print(f" [정답률(Accuracy)]: {acc*100:.2f}%  (높을수록 좋음)")
    # print(f" [곡선 면적(AUC)]  : {auc:.4f}       (1.0에 가까울수록 좋음)")
    print("="*40 + "\n")


    print("[성공] REDCAB_MCAT 학습 및 분석 완료!!!")


    
if __name__ == '__main__':
    main()
