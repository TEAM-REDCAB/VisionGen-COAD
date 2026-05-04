import pandas as pd
import numpy as np
import pickle
import os

# 1. 데이터 로드
df = pd.read_csv('preprocessed_mutation_data.csv')   

# --- 사전(Vocabulary) 구축 ---
# 0은 Padding용으로 남겨두고 1부터 인덱싱합니다.

# 1) Hugo_HGVSc (고유 변이 식별자)
var_vocab = {val: i + 1 for i, val in enumerate(sorted(df['Hugo_HGVSc'].unique()))}

# 2) Variant_Classification (변이 종류)
vc_vocab = {val: i + 1 for i, val in enumerate(sorted(df['Variant_Classification'].unique()))}

# 3) Gene_Function (다중 기능 개별 인덱싱)
all_funcs = set()
df['Gene_Function'].dropna().apply(lambda x: [all_funcs.add(f.strip()) for f in str(x).split(',')])
func_vocab = {val: i + 1 for i, val in enumerate(sorted(list(all_funcs)))}

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
    v_idx = var_vocab.get(row['Hugo_HGVSc'], 0)
    vc_idx = vc_vocab.get(row['Variant_Classification'], 0)
    
    # 쉼표로 연결된 기능을 분리하여 인덱싱 (최대 6개)
    # !! 주의 !! : 기능은 중요도가 없으므로 단순히 등장 순서대로 인덱싱 되었음. 위치정보를 들어가지 않도록 후처리 필요.
    # SNN을 진행하면서 카운트 벡터화 (Multi-Hot Encoding)를 진행방식을 채택함으로써 위치정보는 들어가지않으나 특성은 유지되도록 함.
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

# 1. 어레이 생성 시 사용했던 환자 리스트를 그대로 가져옵니다.
patient_list = df['patient_nm'].unique().tolist()

# 2. 인코딩 상태 파일(encoding_states.pkl)에 환자 명단을 추가로 저장합니다.
with open('genomic_encoding_states.pkl', 'rb') as f:
    states = pickle.load(f)

states['patient_list'] = patient_list # 환자 순서 정보 추가

with open('genomic_encoding_states.pkl', 'wb') as f:
    pickle.dump(states, f)

print(f"환자 {len(patient_list)}명의 순서 정보가 저장되었습니다.")