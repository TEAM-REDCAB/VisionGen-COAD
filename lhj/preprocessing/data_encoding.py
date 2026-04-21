import pandas as pd
import numpy as np
import pickle
import os

# 1. 데이터 로드
df = pd.read_csv('preprocessed_mutation_data.csv')

# --- 사전(Vocabulary) 구축 ---
# 0은 Padding용으로 남겨두고 1부터 인덱싱합니다.

# 1) Hugo_HGVSp (고유 변이 식별자)
var_vocab = {val: i + 1 for i, val in enumerate(sorted(df['Hugo_HGVSp'].unique()))}

# 2) Variant_Classification (변이 종류)
vc_vocab = {val: i + 1 for i, val in enumerate(sorted(df['Variant_Classification'].unique()))}

# 3) Gene_Function (다중 기능 개별 인덱싱)
all_funcs = set()
df['Gene_Function'].dropna().apply(lambda x: [all_funcs.add(f.strip()) for f in str(x).split(',')])
func_vocab = {val: i + 1 for i, val in enumerate(sorted(list(all_funcs)))}

# 기초 통계량 결과
#count     266.000000
#mean       92.477444
#std       185.707605
#min         7.000000
#25%        15.000000
#50%        21.000000
#75%        41.000000
#max      1425.000000

# --- 인코딩 상태 저장 (Pickle) ---
# 나중에 테스트 데이터나 새로운 데이터를 인코딩할 때 이 사전이 반드시 필요합니다.
encoding_states = {
    'var_vocab': var_vocab,
    'vc_vocab': vc_vocab,
    'func_vocab': func_vocab,
    'max_nodes': 1425,
    'max_func_per_node': 6
}

with open('genomic_encoding_states.pkl', 'wb') as f:
    pickle.dump(encoding_states, f)
print("인코딩 사전이 'genomic_encoding_states.pkl'로 저장되었습니다.")

# --- 수치 변환 로직 ---

MAX_NODES = 1425
MAX_FUNC_PER_NODE = 6
# 특징 구성: [Var_ID, VC_ID, F1, F2, F3, F4, F5, F6, VAF] -> 총 9개 채널
FEATURE_SIZE = 1 + 1 + MAX_FUNC_PER_NODE + 1 

def encode_row(row):
    v_idx = var_vocab.get(row['Hugo_HGVSp'], 0)
    vc_idx = vc_vocab.get(row['Variant_Classification'], 0)
    
    # 쉼표로 연결된 기능을 분리하여 인덱싱 (최대 6개)
    # !! 주의 !! : 기능은 중요도가 없으므로 단순히 등장 순서대로 인덱싱 되었음.
    # 살제 모델에서 사용 시 6개 채널 각각 임베딩 벡터를 뽑은 뒤, 이를 더하거나(Sum) 평균(Mean)을 내는 방식으로 사용해서 위치정보를 제거해야함.
    funcs = [f.strip() for f in str(row['Gene_Function']).split(',')]
    f_indices = [func_vocab.get(f, 0) for f in funcs if f in func_vocab]
    
    # 6개 채널에 맞춰 패딩 또는 절단
    if len(f_indices) < MAX_FUNC_PER_NODE:
        f_indices += [0] * (MAX_FUNC_PER_NODE - len(f_indices))
    else:
        f_indices = f_indices[:MAX_FUNC_PER_NODE]
        
    return [v_idx, vc_idx] + f_indices + [row['t_vaf']]

# --- 환자별 데이터 루프 및 3차원 배열 생성 ---

patient_names = df['patient_nm'].unique()
final_matrix = []

for name in patient_names:
    p_df = df[df['patient_nm'] == name]
    
    # 해당 환자의 모든 행 인코딩
    p_nodes = [encode_row(row) for _, row in p_df.iterrows()]
    
    # 1425개 노드 규격에 맞춰 패딩
    actual_count = len(p_nodes)
    if actual_count < MAX_NODES:
        padding = [[0] * FEATURE_SIZE] * (MAX_NODES - actual_count)
        p_nodes += padding
    else:
        p_nodes = p_nodes[:MAX_NODES]
        
    final_matrix.append(p_nodes)

# 넘파이 배열로 변환 (메모리 효율을 위해 float32 사용)
final_matrix = np.array(final_matrix, dtype=np.float32)

# 최종 배열 저장
np.save('genomic_input_matrix.npy', final_matrix)
print(f"최종 배열 형태: {final_matrix.shape}") # (환자수, 1425, 9)
print("'genomic_input_matrix.npy' 파일로 저장이 완료되었습니다.")