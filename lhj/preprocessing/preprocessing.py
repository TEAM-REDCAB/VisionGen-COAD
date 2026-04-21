import pandas as pd
import numpy as np

current_path = "/home/team1/data/TCGA_COAD_multimodal_data"
df = pd.read_csv(current_path + "/TCGA_COAD.wxs.common_parsed.maf", sep="\t", low_memory=False)

# 1. 필터링 및 선택 컬럼 정의 (Gene_Function 추가)
mt_filter = ["Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
             "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation", "Splice_Site"]

select_col = ["Hugo_Symbol", "Variant_Classification", "HGVSp_Short", 
              "t_depth", "t_alt_count", "patient_nm", "Gene_Function"]

# 2. signatures.csv 처리: 유전자별로 기능을 ","로 합친 문자열 딕셔너리 생성
df_sig = pd.read_csv('signatures.csv')
gene_to_func_str = {}

for category in df_sig.columns:
    genes = df_sig[category].dropna().unique()
    for gene in genes:
        if gene not in gene_to_func_str:
            gene_to_func_str[gene] = []
        gene_to_func_str[gene].append(category)

# 리스트를 ", "로 연결된 문자열로 미리 변환
gene_to_func_str = {gene: ", ".join(funcs) for gene, funcs in gene_to_func_str.items()}

# 3. 매핑 (리스트가 아닌 문자열이 바로 들어감)
df['Gene_Function'] = df['Hugo_Symbol'].map(lambda x: gene_to_func_str.get(x, 'Unknown'))

def preprocess(df):
    # 전처리 과정에서 Gene_Function이 유실되지 않도록 select_col 사용
    df1 = df[select_col].copy()
    
    # 대소문자 구분을 위해 mt_filter와 맞는지 확인 후 필터링
    df1 = df1[df1["Variant_Classification"].isin(mt_filter)]
    
    # 결측치 제거 및 VAF 계산
    df1 = df1.dropna()
    df1 = df1[df1['t_depth'] > 0] # 0으로 나누기 방지
    df1['t_vaf'] = df1['t_alt_count'] / df1['t_depth']
    
    # Hugo_Symbol + HGVSp_Short을 결합하여 새로운 컬럼 생성
    df1['Hugo_HGVSp'] = df1['Hugo_Symbol'] + "_" + df1['HGVSp_Short']
    
    # 불필요한 컬럼 제거 및 인덱스 초기화
    df1 = df1.drop(columns=['t_depth', 't_alt_count', "Hugo_Symbol", "HGVSp_Short"])
    df1 = df1.reset_index(drop=True)
    df1 = df1[df1['Gene_Function'] != 'Unknown']
    
    df1 = df1[["patient_nm", "Hugo_HGVSp", "Variant_Classification", "Gene_Function", "t_vaf"]]
    return df1

df_preprocessed = preprocess(df)

# 결과 확인
print(df_preprocessed)

# 전처리가 완료된 df_preprocessed를 사용합니다.
# patient_nm 컬럼을 기준으로 그룹화하여 행의 개수를 셉니다.

patient_counts = df_preprocessed.groupby('patient_nm').size().reset_index(name='mutation_count')

# 개수가 많은 순서대로 정렬
patient_counts = patient_counts.sort_values(by='mutation_count', ascending=False)

# 결과 확인
print(patient_counts)

# 기초 통계량 확인 (평균 변이 수, 최대/최소 등)
print(patient_counts['mutation_count'].describe())

df_preprocessed.to_csv(current_path + '/preprocessed_mutation_data.csv', index=False)