"""
datasets/dataset_classification.py
====================================
TCGA-COAD MSI/MSS 이진 분류용 데이터셋

Genomic 데이터 로딩 방식:
  - genomic_input_matrix.npy  : shape (N_patients, 1425, 9)
      dim[0]   : Variant_ID (encoded)
      dim[1]   : VC_ID
      dim[2:8] : Func_IDs[0~5]  →  6개 omic 그룹 마스크
      dim[8]   : t_vaf           →  피처 값
  - genomic_encoding_states.pkl : 환자 순서(patient_order) 및 인코딩 상태

CSV 최소 필수 컬럼:
  case_id | slide_id | msi_status
  (genomic 컬럼 불필요 — npy에서 직접 로드)
"""

from __future__ import print_function, division
import os
import pickle

import h5py
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════
# 유틸: genomic .npy + .pkl 로드 및 omic 그룹 마스크 계산
# ══════════════════════════════════════════════════════════════════════

def load_genomic_data(genomic_dir: str):
    """
    Parameters
    ----------
    genomic_dir : genomic_input_matrix.npy와 genomic_encoding_states.pkl이
                  있는 폴더 경로

    Returns
    -------
    matrix       : np.ndarray  shape (N, 1425, 9)
    patient_order: list[str]   길이 N, matrix row 순서와 1:1 대응
    omic_masks   : list[np.ndarray]  길이 6, 각 원소 shape (1425,) bool
    omic_sizes   : list[int]   각 omic 그룹의 피처 수
    """
    npy_path = os.path.join(genomic_dir, "genomic_input_matrix.npy")
    pkl_path = os.path.join(genomic_dir, "genomic_encoding_states.pkl")

    assert os.path.exists(npy_path), f"npy 없음: {npy_path}"
    assert os.path.exists(pkl_path), f"pkl 없음: {pkl_path}"

    matrix = np.load(npy_path)                  # (N, 1425, 9)
    with open(pkl_path, "rb") as f:
        enc_states = pickle.load(f)

    # pkl 키 이름 fallback 처리
    if "patient_order" in enc_states:
        patient_order = enc_states["patient_order"]
    elif "patients" in enc_states:
        patient_order = enc_states["patients"]
    elif "patient_list" in enc_states:
        patient_order = enc_states["patient_list"]
    else:
        raise KeyError(
            f"pkl에 patient_order 키 없음. 실제 키: {list(enc_states.keys())}"
        )

    assert matrix.shape[0] == len(patient_order), (
        f"npy 환자 수 {matrix.shape[0]} != pkl 환자 수 {len(patient_order)}"
    )

   # ── omic 마스크 수정 ──────────────────────────────────────────────
    # [기존 문제]
    # group_nonzero[:, i] → F슬롯 i번째가 non-zero인 위치를 omic i로 잘못 해석
    # 실제 npy: dim[2:8] = [F1, F2, F3, F4, F5, F6]
    #           각 슬롯에는 func_vocab 인덱스(1~6)가 순서 없이 채워짐
    #
    # [수정] func_vocab은 알파벳 정렬로 생성됨:
    #   1 → Cell Differentiation Markers
    #   2 → Cytokines and Growth Factors
    #   3 → Oncogenes
    #   4 → Protein Kinases
    #   5 → Transcription Factors
    #   6 → Tumor Suppressor Genes
    #
    # signatures.csv 컬럼 순서(= omic 그룹 순서):
    #   omic0: Tumor Suppressor Genes       → func_idx 6
    #   omic1: Oncogenes                    → func_idx 3
    #   omic2: Protein Kinases              → func_idx 4
    #   omic3: Cell Differentiation Markers → func_idx 1
    #   omic4: Transcription Factors        → func_idx 5
    #   omic5: Cytokines and Growth Factors → func_idx 2

    OMIC_FUNC_INDICES = [6, 3, 4, 1, 5, 2]  # omic0 ~ omic5에 대응하는 func_idx

    func_ids_all = matrix[:, :, 2:8]  # (N, 1425, 6)

    omic_masks = []
    for func_idx in OMIC_FUNC_INDICES:
        # F1~F6 슬롯 중 어디든 해당 func_idx가 있으면 이 그룹 소속
        belongs = (func_ids_all == func_idx).any(axis=2)  # (N, 1425)
        # 전체 환자 중 한 명이라도 해당 위치에 이 그룹 있으면 마스크 True
        group_mask = belongs.any(axis=0)  # (1425,)
        omic_masks.append(group_mask)

    omic_sizes = [int(m.sum()) for m in omic_masks]

    print(f"[Genomic] shape      : {matrix.shape}")
    print(f"[Genomic] patients   : {len(patient_order)}")
    print(f"[Genomic] omic_sizes : {omic_sizes}")
    for i, sz in enumerate(omic_sizes):
        if sz == 0:
            print(f"  ⚠️  omic{i} 그룹에 변이 없음 — Func_IDs 확인 필요")

    return matrix, patient_order, omic_masks, omic_sizes


# ══════════════════════════════════════════════════════════════════════
class Generic_WSI_Classification_Dataset(Dataset):

    def __init__(self,
        csv_path     = "./data/tcga_coad_all_clean.csv",
        genomic_dir  = "./data/genomic",
        mode         = "coattn",
        apply_sig    = False,
        shuffle      = False,
        seed         = 7,
        print_info   = True,
        patient_strat= False,
        label_col    = "msi_status",
        label_dict   = {"MSS": 0, "MSI-H": 1},
    ):
        self.seed          = seed
        self.print_info    = print_info
        self.patient_strat = patient_strat
        self.data_dir      = None
        self.genomic_dir   = genomic_dir
        self.mode          = mode

        # ── 1. CSV 로드 ──────────────────────────────────────────────
        slide_data = pd.read_csv(csv_path, low_memory=False)

        if "case_id" not in slide_data.columns:
            slide_data.index = slide_data.index.str[:12]
            slide_data["case_id"] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        self.label_col = label_col
        assert label_col in slide_data.columns, \
            f"'{label_col}' 컬럼 없음. 컬럼 목록: {slide_data.columns.tolist()}"

        # oncotree_code 필터 (없으면 스킵)
        if "oncotree_code" in slide_data.columns and \
                "IDC" in slide_data["oncotree_code"].values:
            slide_data = slide_data[slide_data["oncotree_code"] == "IDC"]

        # ── 2. 라벨 매핑 ─────────────────────────────────────────────
        slide_data = slide_data.dropna(subset=[label_col])
        slide_data["label"] = slide_data[label_col].map(label_dict)
        slide_data = slide_data.dropna(subset=["label"])
        slide_data["label"] = slide_data["label"].astype(int)

        self.label_dict  = label_dict
        self.num_classes = len(set(label_dict.values()))

        # ── 3. 환자 단위 딕셔너리 ────────────────────────────────────
        patients_df    = slide_data.drop_duplicates(["case_id"]).copy()
        slide_data_idx = slide_data.set_index("case_id")
        patient_dict   = {}
        for pid in patients_df["case_id"]:
            sids = slide_data_idx.loc[pid, "slide_id"]
            patient_dict[pid] = (
                np.array([sids]) if isinstance(sids, str) else sids.values
            )
        self.patient_dict = patient_dict

        slide_data = patients_df.reset_index(drop=True)
        slide_data = slide_data.assign(slide_id=slide_data["case_id"])
        self.patient_data = {
            "case_id": slide_data["case_id"].values,
            "label":   slide_data["label"].values,
        }
        self.slide_data = slide_data

        base_meta     = ["case_id", "slide_id", label_col, "label", "oncotree_code"]
        self.metadata = [c for c in base_meta if c in slide_data.columns]

        self.cls_ids_prep()

        # ── 4. Genomic 배열 로드 ─────────────────────────────────────
        if genomic_dir is not None:
            (self.genomic_matrix,
             self.patient_order,
             self.omic_masks,
             self.omic_sizes) = load_genomic_data(genomic_dir)

            # case_id 앞 12자리 → npy row index 매핑
            self.pid_to_gidx = {
                p[:12]: i for i, p in enumerate(self.patient_order)
            }
        else:
            self.genomic_matrix = None
            self.omic_masks     = [None] * 6
            self.omic_sizes     = [0]    * 6
            self.pid_to_gidx    = {}

        if print_info:
            self.summarize()

    # ──────────────────────────────────────────────────────────────────
    def cls_ids_prep(self):
        self.patient_cls_ids = [
            np.where(self.patient_data["label"] == i)[0]
            for i in range(self.num_classes)
        ]
        self.slide_cls_ids = [
            np.where(self.slide_data["label"] == i)[0]
            for i in range(self.num_classes)
        ]

    def getlabel(self, idx):
        return self.slide_data["label"].iloc[idx]

    def __len__(self):
        return (len(self.patient_data["case_id"]) if self.patient_strat
                else len(self.slide_data))

    def summarize(self):
        print(f"label col    : {self.label_col}")
        print(f"label dict   : {self.label_dict}")
        print(f"num_classes  : {self.num_classes}")
        print("slide counts :\n", self.slide_data["label"].value_counts(sort=False))

    # ──────────────────────────────────────────────────────────────────
    def _get_omic_tensors(self, case_id: str):
        """
        case_id → tuple of 6 FloatTensors (omic0 ~ omic5)

        각 omic_i 텐서 = 해당 그룹 변이들의 t_vaf 벡터
          - 환자가 그 변이를 가지면 → t_vaf 값
          - 없으면 → 0.0 (npy에서 이미 0 패딩됨)
        """
        key = case_id[:12]
        if self.genomic_matrix is None or key not in self.pid_to_gidx:
            return tuple(
                torch.zeros(max(sz, 1), dtype=torch.float32)
                for sz in self.omic_sizes
            )

        gidx  = self.pid_to_gidx[key]
        row   = self.genomic_matrix[gidx]        # (1425, 9)
        t_vaf = row[:, 8].astype(np.float32)     # (1425,)

        return tuple(
            torch.from_numpy(t_vaf[mask])
            for mask in self.omic_masks
        )

    # ──────────────────────────────────────────────────────────────────
    def get_split_from_df(self, all_splits, split_key="train"):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) == 0:
            return None
        mask     = self.slide_data["slide_id"].isin(split.tolist())
        df_slice = self.slide_data[mask].reset_index(drop=True)
        return Generic_Split(
            slide_data     = df_slice,
            metadata       = self.metadata,
            mode           = self.mode,
            data_dir       = self.data_dir,
            label_col      = self.label_col,
            patient_dict   = self.patient_dict,
            num_classes    = self.num_classes,
            genomic_matrix = self.genomic_matrix,
            omic_masks     = self.omic_masks,
            omic_sizes     = self.omic_sizes,
            pid_to_gidx    = self.pid_to_gidx,
            use_h5         = self.use_h5,
        )

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            raise NotImplementedError
        assert csv_path
        all_splits  = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits, "train")
        val_split   = self.get_split_from_df(all_splits, "val")
        return train_split, val_split


# ══════════════════════════════════════════════════════════════════════
class Generic_MIL_Classification_Dataset(Generic_WSI_Classification_Dataset):

    def __init__(self, data_dir, mode="coattn", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode     = mode
        self.use_h5   = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id   = self.slide_data["case_id"].iloc[idx]
        label     = int(self.slide_data["label"].iloc[idx])
        slide_ids = self.patient_dict[case_id]

        data_dir = (
            self.data_dir[self.slide_data["oncotree_code"].iloc[idx]]
            if isinstance(self.data_dir, dict) else self.data_dir
        )

        if not self.data_dir:
            return slide_ids, label

        # ── WSI H5/PT 로드 ───────────────────────────────────────────
        def load_wsi(slide_ids, base_dir):
            bags = []
            for sid in slide_ids:
                sid = sid.rstrip(".svs")
                if not self.use_h5:
                    bags.append(torch.load(
                        os.path.join(base_dir, "pt_files", f"{sid}.pt")))
                else:
                    with h5py.File(
                        os.path.join(base_dir, f"{sid}.h5"), "r"
                    ) as f:
                        bags.append(torch.from_numpy(f["features"][:]))
            return torch.cat(bags, dim=0) if bags else torch.zeros((1, 1536))

        if self.mode == "path":
            return (load_wsi(slide_ids, data_dir), torch.zeros((1, 1)), label)

        elif self.mode == "coattn":
            path_feat = load_wsi(slide_ids, data_dir)
            omics     = self._get_omic_tensors(case_id)   # tuple of 6
            return (path_feat, *omics, label)              # 8-tuple

        elif self.mode == "omic":
            omics = self._get_omic_tensors(case_id)
            return (torch.zeros((1, 1)), torch.cat(omics), label)

        else:
            raise NotImplementedError(f"mode '{self.mode}' not implemented.")


# ══════════════════════════════════════════════════════════════════════
class Generic_Split(Generic_MIL_Classification_Dataset):

    def __init__(self, slide_data, metadata, mode,
                 data_dir=None, label_col=None, patient_dict=None,
                 num_classes=2,
                 genomic_matrix=None, omic_masks=None,
                 omic_sizes=None, pid_to_gidx=None, use_h5=False):

   
        self.slide_data     = slide_data
        self.metadata       = metadata
        self.mode           = mode
        self.data_dir       = data_dir
        self.num_classes    = num_classes
        self.label_col      = label_col
        self.patient_dict   = patient_dict
        # genomic 공유 객체 (부모에서 전달 — 재로드 없음)
        self.genomic_matrix = genomic_matrix
        self.omic_masks     = omic_masks  if omic_masks  is not None else [None]*6
        self.omic_sizes     = omic_sizes  if omic_sizes  is not None else [0]*6
        self.pid_to_gidx    = pid_to_gidx if pid_to_gidx is not None else {}
        self.use_h5         = use_h5

        self.slide_cls_ids = [
            np.where(self.slide_data["label"] == i)[0]
            for i in range(self.num_classes)
        ]

    def getlabel(self, idx):
        return self.slide_data["label"].iloc[idx]

    def __len__(self):
        return len(self.slide_data)

def save_splits(datasets, column_keys, filename, boolean_style=False):
    splits = [d.slide_data['case_id'] for d in datasets if d is not None]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys[:len(splits)]
    else:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys[:len(splits)]
    df.to_csv(filename, index=False)