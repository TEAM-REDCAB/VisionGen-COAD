import numpy as np

# 데이터 로드
matrix = np.load('genomic_input_matrix.npy')

# 첫 번째 환자의 상위 5개 변이 추출 (5, 9)
sample_nodes = matrix[0, 0:5, :]

print("--- [어레이 형태 확인] ---")
print(sample_nodes) 
print("-" * 60)

# 각 행(변이)별로 데이터 의미 해석
for i in range(len(sample_nodes)):
    node = sample_nodes[i]
    print(f"변이 {i+1} 정보:")
    print(f"  - Variant_ID: {node[0]:.0f}")
    print(f"  - VC_ID:      {node[1]:.0f}")
    print(f"  - Func_IDs:   {node[2:8].astype(int)}") # 6개 채널
    print(f"  - t_vaf:      {node[8]:.4f}")
    print("-" * 30)
    
#변이 3 정보:
#  - Variant_ID: 13133
#  - VC_ID:      5
#  - Func_IDs:   [1 0 0 0 0 0]
#  - t_vaf:      0.1473
#------------------------------
#변이 4 정보:
#  - Variant_ID: 12725
#  - VC_ID:      5
#  - Func_IDs:   [5 0 0 0 0 0]
#  - t_vaf:      0.0603
#------------------------------
#변이 5 정보:
#  - Variant_ID: 4605
#  - VC_ID:      6
#  - Func_IDs:   [4 0 0 0 0 0]
#  - t_vaf:      0.2632
#------------------------------