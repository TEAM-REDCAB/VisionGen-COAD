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

