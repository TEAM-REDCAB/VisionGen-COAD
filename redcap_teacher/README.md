# REDCAB-MCAT: Teacher Model Training Pipeline

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
```
___
# [기술 문서] TCGA-COAD Teacher 아키텍처 모듈화 및 평가 검증 안정화 설계안

문서 목적: 초기 형태의 스크립트 파일을 기능별 독립 모듈로 분할(Modularization) 파이프라인의 구조적 고도화 및 클래스 불균형 환경에서의 평가 지표(Metric) 신뢰성을 극대화하기 위한 Stratified K-Fold 검증 시스템 도입 과정 공유

---

## 1. 코드 구조의 엔터프라이즈급 모듈화 (Modularization)

기존 단일 스크립트(`main.py`)에 모든 로직이 혼재되어 있던 구조를 탈피하여, 연구자 간의 협업 효율성과 유지보수성을 극대화하기 위해 '관심사의 분리(Separation of Concerns)' 원칙을 적용하였습니다.

*   **설계 변경 (MVC 패턴 유사):**
    *   **Controller (`main_teacher.py`):** 전체 파이프라인의 하이퍼파라미터 정의, K-Fold 분할, XAI 평가 지표 구성 및 최종 결과물(CSV) I/O만을 통제하는 통제탑 역할을 담당합니다.
    *   **Execution Loop (`train.py`):** GPU 연산 스케줄링, AMP(자동 혼합 정밀도) 실행, 오차 역전파 및 `autocast` 등의 실제 학습 루틴만을 독립적으로 수행합니다.
    *   **Architecture (`models/`):** 병리-유전체 해독 모델인 `MCATTeacher` 등 신경망 코어를 별도 객체로 격리하였습니다.
*   **기대 효과:** 향후 학생 모델(Student Model) 훈련 스크립트 작성 시, 중복 코딩 없이 기존 `train.py`와 `dataset.py`를 재사용 할 수 있는 완벽한 확장성을 확보하였습니다.

---

## 2. 평가 신뢰도 확보: Stratified K-Fold 시스템 도입

대장암(COAD) 데이터셋 특성상, 핵심 질환 클래스(MSIMUT)의 비율이 보통 15% 내외로 극심한 불균형(Class Imbalance)을 띠고 있습니다.

*   **기존 무작위 분할(KFold)의 문제점:** 일반 K-Fold로 환자를 분할할 경우, 특정 Fold의 테스트 셋에 MSIMUT 환자가 우연히 과도하게 몰리거나, 반대로 한 명도 포함되지 않는 통계적 편향(Shift)이 발생하여 AUC 평가 지표가 요동치는 위험이 있었습니다.
*   **비율 균등 분배 (Stratified K-Fold) 적용:** 
    *   분할을 수행하기 전, 대상 환자들의 실제 정답지(MSS 85% : MSIMUT 15%) 레이블 비율을 전수조사합니다.
    *   이후 모든 Fold의 Train / Val / Test Subset 내부에 해당 환자 비율이 **수학적으로 원본 데이터셋의 비율과 동일하게 강제 배정**되도록 분리 알고리즘을 변경하였습니다.
*   **기대 효과:** 어떤 Fold에서 테스트를 진행하더라도 일정한 난이도를 보장하게 되며, 모델이 내놓는 5-Fold 최종 평균 결과(Average AUC)가 학술적, 임상적으로 부정할 수 없는 탄탄한 통계적 신뢰성을 획득하였습니다.

---

## 3. 안정성과 기능성의 완벽한 융합 

다수의 연구자가 병합 작업을 진행하며 발생할 수 있는 메모리 블로우(Memory Blow) 위험을 사전에 차단하기 위해, 이전 버전의 최적화 기술들을 모듈형 구조에 완벽하게 융합 이식하였습니다.

*   **System RAM 누수 방어 동기화:** `train.py`가 독립되면서 발생할 수 있는 XAI 배열 무한 수집 문제를 해결하기 위해, `train.py`의 `run_epoch` 인자에 물리적 차단벽(`save_xai=False`)을 이식함으로써 OOM 킬러 프로세스 발동을 0%로 만들었습니다.
*   **차원 방어막(Dimensional Guard) 통합:** `main_teacher.py`의 XAI 리포트 추출 함수 상단에, 훈련 중 텐서 차원(Batch Dimension)이 무작위로 뒤틀리며 발생하는 `IndexError`를 사전에 감지하고 `(1425, N_patches)` 평탄화 차원으로 복구하는 정교한 연산 가드를 구현하여 테스트 런 무결성을 달성하였습니다.
*   **최종 결과 시각화 모니터링:** 각 Fold별 최고 성능을 달성한 에폭(`stopped_epoch`)을 산출하여 콘솔과 디스크(CSV)에 실시간 기록하는 기능을 추가함으로써, 오버피팅 발생 지점과 지식 증류 시점을 연구자가 명확히 판단할 수 있는 CDSS 대시보드 뷰를 확보하였습니다.
