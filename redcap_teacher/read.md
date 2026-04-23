project/
├── main_teacher.py          ← 실행 진입점 (5-Fold CV)
├── train.py                 ← run_epoch() 공통 루프
├── dataset.py               ← 데이터 로더
├── models/
│   └── mcat_teacher.py      ← 4개 모듈 조립
└── modules/
    ├── path_encoder.py      ← WSI (N, 1536) → (N, 256)
    ├── genomic_encoder.py   ← 유전체 (1425, 9) → (1425, 256) + SNN_Block 인라인
    ├── coattn_fusion.py     ← Co-Attention 핵심 + Attn_Net_Gated   인라인
    └── classifier_head.py   ← (256,) → logits (1, 2)