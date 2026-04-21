# 폴더 및 파일 설명

## raw folder (전처리 전 데이터)
### TCGA_COAD.wxs.common_parsed.maf
* TCGA_COAD Whole Exome Sequencing(WES) Somatic mutation 데이터를 가공함.
* kaggle로 부터 얻은 MSS, MSI의 라벨링이 완료된 환자리스트와 매칭할 수 있는 데이터.

### signatures.csv
* MCAT : https://github.com/mahmoodlab/mcat 에서 사용된 gene function classification 파일

### common_patients.txt
* 환자 별 MSS, SMI 라벨링 리스트

## preprocessing folder (전처리 데이터)
### preprocessed_mutation_data.csv
* TCGA_COAD.wxs.common_parsed.maf 에서 최소한의 컬럼을 선정하고 signatures.csv 를 annotation 한 데이터
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

