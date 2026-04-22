import subprocess

# --- [설정 구간] ---
PATCH_ENCODER = 'gigapath'    # 패치 인코딩 모델(uni_v2, gigapath 등)                 
WSI_FLAT_DIR = './working/flat_wsis/svs'          # 이름이 바뀐 SVS 파일이 모일 곳
RESULT_DIR = './working/trident_processed'    

def run_command(cmd):
    subprocess.run(cmd, shell=True, check=True)

# STEP 3: TRIDENT 전처리 (평탄화된 폴더를 소스로 사용)
trident_cmd = (
    f"python ./TRIDENT/run_batch_of_slides.py --task all "
    f"--wsi_dir {WSI_FLAT_DIR} "
    f"--job_dir {RESULT_DIR} "
    f"--patch_encoder {PATCH_ENCODER} "
    f"--batch_size 32 "
    # 추천하는 TRIDENT 명령어 옵션 조합
    f"--segmenter grandqc --remove_penmarks "
    f"--mag 20 --patch_size 256"
)
run_command(trident_cmd)


print("\n🎉 All tasks finished.")