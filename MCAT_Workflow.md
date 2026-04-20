# Multimodal Workflow (CLAM & MCAT)

## 🛠️ 1단계: 데이터 전처리 (Data Preprocessing)
> 기가픽셀 이미지를 AI가 처리 가능한 특징 벡터 세트($H_{bag}$)로 만드는 "재료 손질" 단계입니다.
> **참고 모델:** [CLAM](https://github.com/mahmoodlab/CLAM.git)

### Step 0. WSI 패치 분할 (Patching)
* **파일:** `CLAM/create_patches_fp.py`
* **내용:** 수십 GB의 WSI 이미지를 배경 없이 조직만 골라 **256x256 크기**로 수만 장 자릅니다.
* **실행:** 원본 슬라이드(`.svs`, `.ndpi` 등)에서 배경을 제거하고 조직 영역만 연속해서 자릅니다.
* **결과:** 각 환자별로 패치들의 좌표값이 담긴 `.h5` 파일이 생성됩니다.

### Step 1. 이미지 특징 추출 (Feature Extraction)
* **파일:** `CLAM/extract_features_fp.py`
* **내용:** 잘린 패치들을 **ResNet-50**에 통과시켜 각 패치당 **1024차원의 특성 벡터(Feature Vector)** 로 변환합니다.
* **결과:** 환자당 하나의 `.pt` 파일이 생성되며, 멀티모달 모델의 이미지 입력값이 됩니다. 형태: `(패치개수, 1024)`
  > 논문에서 지칭하는 $H_{bag}$ 에 해당합니다.

---

## 📥 2단계: 데이터 로드 및 매칭 (Data Loading)
> 이미지 파일과 유전체 CSV 데이터를 환자 ID별로 짝지어 "식탁을 차리는" 단계입니다.
> **참고 모델:** [MCAT](https://github.com/mahmoodlab/MCAT.git)

### Step 2. 멀티모달 데이터셋 구축
* **파일:** `MCAT/datasets/dataset_survival.py`
* **클래스:** `Generic_MIL_Survival_Dataset`
* **준비물:**
  * Step 1에서 만든 **특징 폴더** (`.pt` 파일들)
  * **환자 정보 CSV**: 환자 ID, 생존 시간, 생존 여부, 그리고 수천 개의 **유전자 발현 데이터(Raw Genomic Data)**
* **실행 내용:** 환자 ID를 기준으로 **이미지 특징(.pt)**, **유전체 수치(CSV)**, **생존 정답지**를 한 세트로 묶어 모델에 올려줍니다.
* **주의사항:** `dataset_generic.py`는 기본 틀일 뿐이며, 유전자 데이터를 실제로 읽어오는 핵심 로직은 `dataset_survival.py` 에서 일어납니다.

---

## 🧠 3단계: 멀티모달 모델 설계 (Model Architecture)
> 모델 내부에서 유전체를 쪼개고 이미지와 섞는 "핵심 요리" 단계입니다.
> **주요 파일:** `MCAT/models/model_coattn.py`

### Step 3-1. 유전체 백($G_{bag}$) 구성 (Genomic Bag Construction) ⭐
* **위치:** `MCAT_Surv` 클래스 내 `forward` 함수 초입
* **실행 내용:** 2단계에서 읽어온 긴 원본 유전자 시퀀스를 6개의 **Signature Networks(`self.sig_networks`)** 에 통과시켜 **6가지 생물학적 범주의 임베딩($G_{bag}$)** 으로 쪼개어 구성합니다.

### Step 3-2. 유전체 유도 공동 주의 집중 (GCA Layer Fusion)
* **위치:** `class GenomicGuidedCoAttention`
* **실행 내용:** 구성된 유전체 백($G_{bag}$)을 **Query**로, 이미지 특징($H_{bag}$)을 **Key/Value**로 사용하여 서로의 연관성을 계산(Attention)하고 융합합니다. 결과적으로 수만 개의 이미지 패치 중 **생존과 관련된 시각적 특징만 필터링**하게 됩니다.

### Step 3-3. 세트 기반 MIL 트랜스포머 (Set-based MIL Transformers)
* **위치:** `self.transformer_h`, `self.transformer_g`
* **실행 내용:** 융합된 이미지와 유전체 정보 각각의 관계를 트랜스포머 레이어로 다시 한번 깊게 분석하고, 모든 정보를 최종적으로 압축하여 하나의 벡터로 요약합니다.

---

## 📉 4단계: 학습 및 예측 (Training & Prediction)
> 모델 가동하여 위험도를 맞히고 똑똑하게 만드는 "실행" 단계입니다.

### Step 4. 생존 위험도 산출 및 학습 루프
* **주요 파일:** 
  * 전체 제어: `MCAT/main.py` 
  * 손실 함수: `MCAT/utils/utils.py`
* **실행 내용:** 모델의 마지막 출력층(`self.classifier`)에서 나온 **위험 점수(Risk Score)** 를 바탕으로 환자의 **사망 위험도(Hazard)** 수치를 예측합니다.
* **훈련(Training):** `NLLSurvLoss` (손실 함수)를 사용하여 모델의 예측값과 실제 환자의 생존 시간(및 검열 여부)간의 오차를 계산하고, 이를 줄이는 방향으로 모델 파라미터를 업데이트(가중치 학습)합니다.
