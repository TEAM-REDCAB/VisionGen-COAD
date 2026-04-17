import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# 1. 데이터 로드 (환자 ID와 MSI/MSS 타입만 있는 상태)
df = pd.read_csv('common_patients.txt', sep='\t')
# 가정: df는 'patient'와 'type' (0: MSS, 1: MSI) 컬럼을 가짐
df['type'] = df['type'].map({'MSS':0, 'MSIMUT':1})

# 2. 금고에 넣을 Test Set 20%를 완전히 격리
# stratify=df['type']를 설정하여 Test Set에도 원본 비율과 똑같이 MSI/MSS가 섞이게 함
df_train_val, df_test = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['type'], 
    random_state=42  # 재현성을 위한 Seed 고정
)

# 3. 각 Fold별 컬럼 생성 및 초기화
for i in range(5):
    df[f'fold_{i}'] = 'none'

# 4. Train/Val 데이터(80%) 내에서 Stratified 5-Fold 수행
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# skf.split은 원본의 클래스 비율을 유지하며 인덱스를 반환합니다
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val['type'])):
    
    # 쪼개진 인덱스를 원본 데이터프레임의 실제 인덱스와 매핑
    actual_train_idx = df_train_val.iloc[train_idx].index
    actual_val_idx = df_train_val.iloc[val_idx].index
    
    # 해당 Fold 컬럼에 train, val, test 상태 기록
    df.loc[actual_train_idx, f'fold_{fold_idx}'] = 'train'
    df.loc[actual_val_idx, f'fold_{fold_idx}'] = 'val'
    df.loc[df_test.index, f'fold_{fold_idx}'] = 'test'

# 5. 결과 확인
print(df.head(10))
df.to_csv('clinical_data_folds.csv', index=False)