### **WSI 전처리 자동화**
- extract_features_batch_wsi.py : common_manifest.txt에 있는 환자 슬라이드를 gdc에서 다운로드-조직분류-패치분할-특성추출-캐시삭제 자동화
    - 명령어 : python ./TRIDENT/run_batch_of_slides.py      # 여러 슬라이드 배치 처리
                        --task all                          # 처리할 절차 선택(choices=['seg', 'coords', 'feat', 'all'])
                        --wsi_dir {WSI_FLAT_DIR}            # 배치 처리될 svs 파일이 저장될 폴더 지정
                        --job_dir {RESULT_DIR}              # 결과물이 저장될 폴더 지정
                        --patch_encoder {PATCH_ENCODER}     # 패치 인코딩에 사용할 엔진 모델(uni_v2, gigapath 등)
                        --batch_size 32                     # 패치 분할 및 특성 추출 배치 사이즈(기본:64)
                        --segmenter grandqc                 # 조직 분할에 사용할 모델(choices=['hest', 'grandqc', 'otsu'])
                        --remove_penmarks                   # 펜 자국 제거
                        --mag 20                            # 배율을 x20으로 통일
                        --patch_size 256                    # 추출할 패치 사이즈

- huggingface의 UNI_v2/Prov-GigaPath 사전학습 모델 가중치 사용(huggingface에서 사용 승인 필요)
- H&E 염색에 최적화된 GrandQC 모델을 사용, 펜 자국 제거(--remove_penmarks) -> GPU 필수!
- svs 파일의 메타 정보를 읽어 20배율로 고정한 후 256x256 크기로 패치 분할 
- 슬라이드 단위로 저장(h5형식의 이미지 임베딩 벡터, 좌표 벡터)되며, 추후 멀티모달 학습 시 환자단위 병합 필요
    - 이미지 예시   : TCGA-AA-A01T-01Z-00-DX1.h5            (num_patch, features)       = (3331, 1536)
    - 좌표 예시     : TCGA-AA-A01T-01Z-00-DX1_patches.h5    (num_patch, coords(x, y))   = (3331, 2)
- 배치 단위로 다운로드-특성 추출 후 공간확보를 위해 원본 WSI가 삭제됨
- 저장경로(리눅스 서버) : ~/data/trident_processed


Thumbnail
![thumbnail](images/thumbnail.jpg)

Tissue Segmentation
![segmentation](images/contour.jpg)

Patch Extraction
![patch tiling](images/patch.jpg)

Visualize Attention Score
![attention score](images/attention_score.png)



### **테스트용 ABMIL 이미지 추론 모델**
- abmil_model.py : 랜덤 시드, 정답지 분할, 각종 경로(정답지, 모델 저장, 테스트 결과, 시각화 자료 등), 공통 사용 클래스(dataset, classifier, focal loss) 정의
- abmil_train.py : TRIDENT 내장 Attention-based Multi Instance Learning 5-fold 교차검증 훈련
- abmil_test.py : ABMIL 테스트
- ablmil_visualize_hitmap.py : 어텐션 스코어를 히트맵으로 시각화


### **테스트용 GigaPath 이미지 추론 모델**
- gigapath_model.py : 랜덤 시드, 정답지 분할, 각종 경로(정답지, 모델 저장, 테스트 결과, 시각화 자료 등), 공통 사용 클래스(dataset, classifier, focal loss) 정의
- gigapath_train.py : flash_attn(xformers 대체)를 사용한 경량 ViT 모델로 5-fold(stratified) 교차검증 훈련, Focal Loss(불균형 데이터 가중치) 손실함수 채택
- gigapath_test.py : 저장된 모델 테스트용 코드