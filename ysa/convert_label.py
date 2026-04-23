# convert_label.py 수정본
import os
import pandas as pd

TXT_PATH = '/home/team1/data/coad_msi_mss/TCGA_COAD_WSI_info.txt' 
H5_DIR   = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
OUT_PATH = './data/tcga_coad_all_clean.csv'

# 라벨 로드
label_df = pd.read_csv(TXT_PATH, sep='\t')
type_map = {'MSIMUT': 'MSI', 'MSS': 'MSS'}
label_df['msi_status'] = label_df['type'].map(type_map)
label_dict = dict(zip(label_df['patient'].str[:12], label_df['msi_status']))

# h5 스캔 → 슬라이드별 행 생성 (DX1, DX2 각각 행으로)
h5_files = [f.replace('.h5', '') for f in os.listdir(H5_DIR) if f.endswith('.h5')]

rows, no_label = [], []
for slide_id in h5_files:
    pid = slide_id[:12]
    if pid in label_dict:
        rows.append({
            'case_id':    pid,        # 환자 ID (12자리)
            'slide_id':   slide_id,   # 슬라이드 ID (풀네임)
            'msi_status': label_dict[pid]
        })
    else:
        no_label.append(slide_id)

if no_label:
    print(f"⚠️  라벨 없는 슬라이드 {len(no_label)}개: {no_label}")

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"✅ 저장 완료: {OUT_PATH}")
print(f"슬라이드 수  : {len(df)}개")
print(f"고유 환자 수 : {df['case_id'].nunique()}명")
print(df['msi_status'].value_counts().to_string())