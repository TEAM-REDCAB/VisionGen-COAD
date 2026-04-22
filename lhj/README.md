# 폴더 및 파일 설명

## raw directory (전처리 전 데이터)
### TCGA_COAD.wxs.common_parsed.maf
* TCGA_COAD Whole Exome Sequencing(WES) Somatic mutation 데이터를 가공함.
* kaggle로 부터 얻은 MSS, MSI의 라벨링이 완료된 환자리스트와 매칭할 수 있는 데이터.

### signatures.csv
* MCAT : https://github.com/mahmoodlab/mcat 에서 사용된 gene function classification 파일

### common_patients.txt
* 환자 별 MSS, SMI 라벨링 리스트

### simpleRepeat.tar.gz
* MSIpred (baseline) 모델 성능 테스트를 위한 필수 파일

## preprocessing directory (전처리 데이터)
### preprocessed_mutation_data.csv
* TCGA_COAD.wxs.common_parsed.maf 에서 최소한의 컬럼을 선정하고 signatures.csv 를 annotation 한 데이터
* TCGA_COAD.wxs.common_parsed.maf 에서 사용한 컬럼 및 설명 요약

| Column | Description |
| :--- | :--- |
| **Hugo_Symbol** | HUGO symbol for the gene (HUGO symbols are always in all caps). "Unknown" is used for regions that do not correspond to a gene. |
| **Variant_Classification** | Translational effect of variant allele (e.g., Missense_Mutation, Nonsense_Mutation, etc.). |
| **HGVSc** | The coding sequence of the variant in HGVS recommended format. |
| **t_depth** | Read depth across this locus in tumor BAM. |
| **t_alt_count** | Read depth supporting the variant allele in tumor BAM. |
| **patient_nm** | Patient identification (ID) used for cohort mapping. |

### preprocessing.py
* preprocessed_mutation_data.csv 생성 및 간단한 통계 확인 스크립트
### array_preview.py
* genomic_input_matrix.npy 파일의 예시를 볼 수 있는 스크립트 (예시결과 스크립트 내 포함되어 있음)
### data_encoding.py
* preprocessed_mutation_data.csv로부터 encoding후 array형태로 생성하는 과정의 스크립트
### genomic_encoding_states.pkl
* 인코딩 상태 및 환자 순서 정보
### genomic_input_matrix.npy
* 환자 별 genomic 데이터 정보 array (환자 수, 1425, 9)
* **주의** 데이터 인덱스의 순서가 바뀌면 인코딩 상태에 저장한 환자 명과 비교가 불가. WSI와 매칭할 수 없음. 재작업 진행해야 함.

## baseline directory (유전체데이터 싱글 모델 테스트)
### MSIpred
* result directory : TCGA-COAD데이터를 사용한 툴 실행 결과
* tool directory : 실행 프로그램 디렉토리
### SNN
* result directory : TCGA-COAD데이터를 사용한 SNN model 결과. feature 변경을 진행하며 3가지 테스트 진행
* script directory : SNN baseline 테스트를 위한 스크립트 (base1 = 모든 피쳐를 사용. base2 = 가장 영향력이 큰 피쳐를 제외 함. base3 = 가장 영향력이 큰 피쳐를 변형.)


