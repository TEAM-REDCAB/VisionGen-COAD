import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold

def create_mcat_exact_splits(csv_path="./data/tcga_coad_all_clean.csv", output_dir="./splits"):
    """
    중복된 case_id(다중 슬라이드)를 환자 단위로 묶어서 Data Leakage를 방지하는 분할 코드
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 원본 CSV 로드 (슬라이드 단위)
    df_all = pd.read_csv(csv_path)
    
    # 2. 🚨 핵심 수정: 환자(case_id) 단위로 중복 제거하여 '환자 명단' 만들기
    # 한 환자는 MSI 상태가 1개뿐이므로, case_id와 msi_status만 남기고 중복을 날립니다.
    df_patient = df_all[['case_id', 'msi_status']].drop_duplicates().reset_index(drop=True)
    
    print(f"📊 전체 슬라이드: {len(df_all)}개 ➡️ 실제 환자 수: {len(df_patient)}명 (이 기준으로 나눕니다!)")

    # 3. 환자 기준으로 Test Set (20%) 분리
    df_patient_train_val, df_patient_test = train_test_split(
        df_patient, 
        test_size=0.20, 
        random_state=42, 
        stratify=df_patient['msi_status']
    )

    # 4. Test Set 저장: 평가할 때는 슬라이드 정보가 다 필요하므로, 
    # 원본(df_all)에서 Test로 뽑힌 환자들의 모든 슬라이드를 싹 다 가져와서 저장합니다.
    df_test_full = df_all[df_all['case_id'].isin(df_patient_test['case_id'])]
    test_save_path = os.path.join(output_dir, "test_holdout.csv")
    df_test_full.to_csv(test_save_path, index=False)
    print(f"🔒 [금고 보관] Test Set 환자 {len(df_patient_test)}명의 슬라이드 {len(df_test_full)}개 저장 완료")

    # 5. 남은 환자들(80%)로 MCAT 전용 5-Fold CSV 생성
    print(f"\n🔄 남은 환자 {len(df_patient_train_val)}명으로 5-Fold 분할 시작...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_idx = 0
    # 환자 단위로 5-Fold를 나눕니다!
    for train_index, val_index in skf.split(df_patient_train_val['case_id'], df_patient_train_val['msi_status']):
        
        train_cases = df_patient_train_val.iloc[train_index]['case_id'].values
        val_cases = df_patient_train_val.iloc[val_index]['case_id'].values
        
        # train, val 컬럼에 환자 ID 할당
        fold_df = pd.DataFrame({
            'train': pd.Series(train_cases),
            'val': pd.Series(val_cases)
        })
        
        fold_save_path = os.path.join(output_dir, f"splits_{fold_idx}.csv")
        fold_df.to_csv(fold_save_path, index=False)
        
        print(f"   ✅ Fold {fold_idx} 저장 완료: Train {len(train_cases)}명 | Val {len(val_cases)}명")
        fold_idx += 1

    print("\n🎉 환자 단위 Data Leakage 방지 처리 완벽 적용 끝!")



create_mcat_exact_splits()