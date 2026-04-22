# check_connections.py
import os, pickle
import numpy as np
import pandas as pd
import h5py

GENOMIC_DIR = './data'
H5_DIR      = '/home/team1/data/trident_processed/20.0x_256px_0px_overlap/features_uni_v2'
CSV_PATH    = './data/tcga_coad_all_clean.csv'

print("=" * 50)

# ── 1. NPY 확인 ───────────────────────────────────────
npy_path = os.path.join(GENOMIC_DIR, 'genomic_input_matrix.npy')
print(f"\n[1] NPY: {npy_path}")
if os.path.exists(npy_path):
    mat = np.load(npy_path)
    print(f"    shape : {mat.shape}  (기대값: (N, 1425, 9))")
    print(f"    dtype : {mat.dtype}")
    if mat.ndim == 3 and mat.shape[1] == 1425 and mat.shape[2] == 9:
        print("    ✅ shape 정상")
    else:
        print("    ❌ shape 불일치 — dataset_classification.py 수정 필요")
else:
    print("    ❌ 파일 없음")
    mat = None

# ── 2. PKL 확인 ───────────────────────────────────────
pkl_path = os.path.join(GENOMIC_DIR, 'genomic_encoding_states.pkl')
print(f"\n[2] PKL: {pkl_path}")
po = []
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        enc = pickle.load(f)
    print(f"    keys : {list(enc.keys())}")
    # ★ 수정: fallback 처리 + po 항상 정의
    po = enc.get('patient_order') or enc.get('patients') or []
    if po:
        print(f"    patient_order 길이 : {len(po)}")
        print(f"    샘플 (앞 3개)       : {po[:3]}")
        print("    ✅ patient_order 키 존재")
    else:
        print(f"    ❌ patient_order/patients 키 없음 → KeyError 발생")
else:
    print("    ❌ 파일 없음 → 생성 필요")

# ── 3. NPY ↔ PKL 환자 수 매칭 ────────────────────────
print(f"\n[3] NPY ↔ PKL 환자 수 매칭")
if mat is not None and po:
    if mat.shape[0] == len(po):
        print(f"    ✅ 둘 다 {mat.shape[0]}명으로 일치")
    else:
        print(f"    ❌ npy={mat.shape[0]}명 vs pkl={len(po)}명 불일치")
else:
    print("    ⚠️  npy 또는 pkl 없어서 스킵")

# ── 4. H5 파일 확인 ───────────────────────────────────
print(f"\n[4] H5 파일: {H5_DIR}")
h5_ids = set()
if os.path.isdir(H5_DIR):
    h5_files = [f for f in os.listdir(H5_DIR) if f.endswith('.h5')]
    print(f"    파일 수 : {len(h5_files)}개")
    if h5_files:
        sample = h5_files[0]
        with h5py.File(os.path.join(H5_DIR, sample), 'r') as f:
            keys       = list(f.keys())
            feat_shape = f['features'].shape if 'features' in f else '❌ features 키 없음'
        print(f"    샘플 파일  : {sample}")
        print(f"    h5 keys    : {keys}")
        print(f"    features   : {feat_shape}")
        # ★ UNI v2는 1536차원 — 실제 dim 출력해서 확인
        if isinstance(feat_shape, tuple):
            print(f"    feature dim: {feat_shape[1]}  (CLAM=1024, UNI v2=1536 인지 확인)")
        h5_ids = set(f.replace('.h5', '') for f in h5_files)

    # ★ h5_files 서브폴더 있는지도 확인
    sub = os.path.join(H5_DIR, 'h5_files')
    if os.path.isdir(sub):
        print(f"    ⚠️  h5_files 서브폴더 존재: {sub}")
        print(f"       dataset_classification.py가 base_dir/h5_files/를 보므로")
        print(f"       main.py data_root_dir은 {H5_DIR} 이어야 함")
    else:
        print(f"    ℹ️  h5_files 서브폴더 없음 → h5 파일이 이 폴더에 바로 있음")
        print(f"       → dataset_classification.py load_wsi 경로 수정 필요할 수 있음")
else:
    print("    ❌ 폴더 없음")

# ── 5. CSV 확인 ───────────────────────────────────────
print(f"\n[5] Dataset CSV: {CSV_PATH}")
if CSV_PATH and os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"    shape   : {df.shape}")
    print(f"    columns : {df.columns.tolist()}")
    if 'msi_status' in df.columns:
        print(f"    msi_status 분포:\n{df['msi_status'].value_counts().to_string()}")

    if 'slide_id' in df.columns and h5_ids:
        # ★ 수정: rstrip → replace
        csv_ids  = set(df['slide_id'].str.replace('.svs', '', regex=False))
        matched  = csv_ids & h5_ids
        only_csv = csv_ids - h5_ids
        only_h5  = h5_ids  - csv_ids
        print(f"\n    CSV↔H5 매칭 : {len(matched)}개 일치")
        if only_csv: print(f"    CSV에만 있음 : {len(only_csv)}개  (예: {list(only_csv)[:3]})")
        if only_h5:  print(f"    H5에만 있음  : {len(only_h5)}개  (예: {list(only_h5)[:3]})")

    if po and 'case_id' in df.columns:
        csv_pids  = set(df['case_id'].str[:12])
        pkl_pids  = set(p[:12] for p in po)
        matched_p = csv_pids & pkl_pids
        print(f"\n    CSV↔PKL 환자 매칭 : {len(matched_p)}명 일치")
        only_csv_p = csv_pids - pkl_pids
        if only_csv_p:
            print(f"    CSV에만 있는 환자 : {len(only_csv_p)}명 (genomic 없음 → zero padding 처리됨)")
else:
    print("    ⚠️  CSV 없음 → 더미 생성 필요")

print("\n" + "=" * 50)
print("진단 완료. 위 결과 공유해주세요!")