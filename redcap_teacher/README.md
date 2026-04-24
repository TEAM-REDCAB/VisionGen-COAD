# REDCAP-MCAT: Teacher Model Training Pipeline

이 프로젝트는 병리 이미지(WSI)와 유전체 데이터(Genomic Data)를 결합하여 대장암(COAD)의 MSI/MSS 상태를 예측하는 **MCAT(Multimodal Co-Attention Transformer) Teacher 모델** 학습 파이프라인입니다.

## 실행 코드: python main_teacher.py

## 📌 주요 특징
- **Multimodal Fusion**: Gigapath 기반의 WSI 특징과 유전체 변이 데이터를 Co-Attention 메커니즘으로 융합합니다.
- **Stratified 5-Fold CV**: 데이터 불균형을 고려하여 MSI/MSS 비율을 유지하며 학습 및 검증을 수행합니다.
- **Early Stopping & Best Model Recovery**: Validation Loss를 모니터링하여 최적의 시점에 학습을 중단하고 가중치를 복원합니다.
- **XAI (Explainable AI)**: Co-Attention 스코어를 분석하여 특정 유전체 변이가 어떤 병리 패치에 주목했는지 분석 리포트를 생성합니다.
- **AMP (Automatic Mixed Precision)**: FP16 학습을 통해 GPU 메모리 효율과 속도를 최적화합니다.

## 📂 프로젝트 구조

```text
project/
├── main_teacher.py        # 실행 진입점 (Stratified 5-Fold CV 기반 전체 파이프라인 제어)
├── train.py               # run_epoch(): 학습 및 평가를 위한 공통 루프 로직
├── dataset.py             # Pathology-Genomic 결합 데이터셋 로드 및 전처리
├── models/
│   └── mcat_teacher.py    # 각 모듈을 조립하여 최종 Teacher 모델 아키텍처 정의
└── modules/
    ├── path_encoder.py    # 병리 특징 인코더: WSI (N, 1536) → (N, 256)
    ├── genomic_encoder.py # 유전체 인코더: (1425, 9) → (1425, 256) (SNN_Block 포함)
    ├── coattn_fusion.py   # Co-Attention 기반 상호작용 및 특징 융합 (Attn_Net_Gated 포함)
    └── classifier_head.py # 최종 분류기: 융합된 특징 (256,) → Logits (2)
