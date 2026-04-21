"""
prepare_genomic_features.py
============================
mutation.csv → tcga_coad_all_clean.csv 전처리 스크립트

입력:
  - mutation_csv  : patient_nm, Hugo_HGVSp, Variant_Classification,
                    Gene_Function, Variant_ID, VC_ID, Func_IDs, t_vaf
  - meta_csv      : case_id, slide_id, msi_status (최소 3컬럼)

출력:
  - dataset_csv/tcga_coad_all_clean.csv
      case_id | slide_id | msi_status | v{id}_g0 | v{id}_g1 | ...
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
MUTATION_CSV = "path/to/mutation.csv"       # ← 실제 경로로 변경
META_CSV     = "path/to/meta.csv"           # case_id, slide_id, msi_status
OUT_DIR      = Path("dataset_csv")
OUT_CSV      = OUT_DIR / "tcga_coad_all_clean.csv"

N_OMIC_GROUPS = 6                            # MCAT omic 브랜치 수 (고정)
VAF_FILL      = 0.0                          # 해당 변이 없는 환자의 기본값

# ────────────────────────────────────────────────
# 1. 변이 데이터 로드
# ────────────────────────────────────────────────
mut = pd.read_csv(MUTATION_CSV)

# Func_IDs 파싱: "[1, 0, 0, 0, 0, 0]" 문자열 또는 이미 리스트/배열
def parse_func_ids(val):
    if isinstance(val, str):
        val = val.strip()
        # "[1 0 0 0 0 0]" (공백 구분) 또는 "[1, 0, 0, 0, 0, 0]" (콤마 구분)
        val = val.replace(" ", ",").replace(",,", ",")
        return ast.literal_eval(val)
    return list(val)

mut["Func_IDs_parsed"] = mut["Func_IDs"].apply(parse_func_ids)

# Func_IDs 길이 검증
assert all(len(f) == N_OMIC_GROUPS for f in mut["Func_IDs_parsed"]), \
    f"Func_IDs 길이가 {N_OMIC_GROUPS}이 아닌 행이 있습니다."

# ────────────────────────────────────────────────
# 2. 변이별 omic 그룹 할당
#    Func_IDs[i] != 0 → 해당 변이는 omic group i 소속
#    하나의 변이가 여러 그룹에 동시 소속될 수 있음 (다중 레이블)
# ────────────────────────────────────────────────
records = []
for _, row in mut.iterrows():
    func_ids = row["Func_IDs_parsed"]
    for g_idx, func_val in enumerate(func_ids):
        if func_val != 0:
            records.append({
                "patient_nm": row["patient_nm"],
                "Variant_ID": int(row["Variant_ID"]),
                "omic_group": g_idx,
                "t_vaf":      float(row["t_vaf"]),
                # 중복 변이가 있으면 max VAF 사용 (pivot aggfunc 참조)
            })

assigned = pd.DataFrame(records)
print(f"[1] 변이-그룹 할당 완료: {len(assigned)}개 (원본 {len(mut)}개 변이)")

# 그룹별 변이 수 분포 확인
print("\n[omic group 분포]")
print(assigned.groupby("omic_group")["Variant_ID"].nunique().rename("unique_variants"))

# ────────────────────────────────────────────────
# 3. 피벗: 환자 × (Variant_ID, omic_group) → t_vaf
#    컬럼명: v{Variant_ID}_g{omic_group}
# ────────────────────────────────────────────────
assigned["feat_col"] = (
    "v" + assigned["Variant_ID"].astype(str) +
    "_g" + assigned["omic_group"].astype(str)
)

# 환자 × 피처 행렬 (같은 환자-변이가 여러 row면 max VAF 사용)
pivot = assigned.pivot_table(
    index="patient_nm",
    columns="feat_col",
    values="t_vaf",
    aggfunc="max",      # 중복 시 최대 VAF
    fill_value=VAF_FILL,
)
pivot = pivot.reset_index()
print(f"\n[2] Pivot 완료: {pivot.shape[0]}명 × {pivot.shape[1]-1}개 피처")

# ────────────────────────────────────────────────
# 4. 메타 CSV와 병합
#    patient_nm ↔ case_id 매핑
#    (TCGA barcode가 다를 경우 앞 12자리로 맞춤)
# ────────────────────────────────────────────────
meta = pd.read_csv(META_CSV)

# case_id가 TCGA-XX-XXXX-... 형식이면 앞 12자리 추출해서 키로 사용
meta["join_key"]  = meta["case_id"].str[:12]
pivot["join_key"] = pivot["patient_nm"].str[:12]

merged = meta.merge(pivot.drop(columns=["patient_nm"]),
                    on="join_key", how="left")
merged = merged.drop(columns=["join_key"])

# 피처 NaN (genomic 데이터 없는 환자) → 0 채움
feat_cols = [c for c in merged.columns if c.startswith("v") and "_g" in c]
merged[feat_cols] = merged[feat_cols].fillna(VAF_FILL)

print(f"\n[3] 메타 병합 완료: {merged.shape}")
missing = merged[feat_cols].isnull().any(axis=1).sum()
if missing > 0:
    print(f"    ⚠️  genomic 데이터 없는 환자: {missing}명 → 0으로 채움")

# ────────────────────────────────────────────────
# 5. MSI-L 처리 (TCGA-COAD에 일부 존재)
#    선택 A: 제거 / 선택 B: MSI-H로 통합
# ────────────────────────────────────────────────
before = len(merged)
MSI_L_ACTION = "remove"  # "remove" 또는 "merge_to_msih"

if "msi_status" in merged.columns:
    if MSI_L_ACTION == "remove":
        merged = merged[merged["msi_status"].isin(["MSS", "MSI-H"])]
        print(f"\n[4] MSI-L 제거: {before} → {len(merged)}명")
    elif MSI_L_ACTION == "merge_to_msih":
        merged["msi_status"] = merged["msi_status"].replace("MSI-L", "MSI-H")
        print(f"\n[4] MSI-L → MSI-H 통합")

# ────────────────────────────────────────────────
# 6. 저장
# ────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_CSV, index=False)
print(f"\n✅ 저장 완료: {OUT_CSV}")
print(f"   최종 shape: {merged.shape}")
print(f"   MSS: {(merged['msi_status']=='MSS').sum()}명 | "
      f"MSI-H: {(merged['msi_status']=='MSI-H').sum()}명")

# ────────────────────────────────────────────────
# 7. omic_sizes 미리보기 (dataset_classification.py에서 사용할 값)
# ────────────────────────────────────────────────
print("\n[omic_sizes 미리보기]")
for g in range(N_OMIC_GROUPS):
    cols = [c for c in feat_cols if c.endswith(f"_g{g}")]
    print(f"  omic{g+1}: {len(cols)}개 피처")