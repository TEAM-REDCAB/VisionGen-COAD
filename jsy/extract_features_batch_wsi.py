import os
import pandas as pd
import subprocess
import shutil
import glob

# --- [설정 구간] ---
MANIFEST_PATH = './common_manifest.txt' 
GDC_CLIENT_PATH = './gdc-client'     
WSI_TEMP_DIR = './temp_wsis'          # UUID 폴더들이 담길 곳
WSI_FLAT_DIR = './flat_wsis'          # 이름이 바뀐 SVS 파일이 모일 곳
RESULT_DIR = './trident_processed'    
BATCH_SIZE = 5       

def run_command(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 초기 폴더 생성
os.makedirs(WSI_TEMP_DIR, exist_ok=True)
os.makedirs(WSI_FLAT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_csv(MANIFEST_PATH, sep='\t')
total_slides = len(df)

for i in range(0, total_slides, BATCH_SIZE):
    batch_df = df.iloc[i : i + BATCH_SIZE]
    batch_manifest_path = f'tmp_batch_{i}.txt'
    batch_df.to_csv(batch_manifest_path, sep='\t', index=False)
    
    print(f"\n--- [Batch {i//BATCH_SIZE + 1}] Processing ---")

    try:
        # STEP 1: GDC Download
        run_command(f"{GDC_CLIENT_PATH} download -m {batch_manifest_path} -d {WSI_TEMP_DIR}")

        # STEP 2: 폴더 구조 평탄화 및 파일명 변경 (UUID 제거)
        print("📂 Flattening and Renaming SVS files...")
        svs_files = glob.glob(os.path.join(WSI_TEMP_DIR, "**/*.svs"), recursive=True)
        
        for old_path in svs_files:
            file_name = os.path.basename(old_path)
            # 요청하신 대로 첫 번째 '.'을 기준으로 자릅니다.
            # TCGA-AA-3939-01Z-00-DX1.uuid.svs -> TCGA-AA-3939-01Z-00-DX1.svs
            new_base_name = file_name.split('.')[0]
            new_path = os.path.join(WSI_FLAT_DIR, f"{new_base_name}.svs")
            
            shutil.move(old_path, new_path)
            print(f"   - {file_name} -> {new_base_name}.svs")

        # STEP 3: TRIDENT 전처리 (평탄화된 폴더를 소스로 사용)
        trident_cmd = (
            f"python ./TRIDENT/run_batch_of_slides.py --task all "
            f"--wsi_dir {WSI_FLAT_DIR} "
            f"--job_dir {RESULT_DIR} "
            f"--patch_encoder uni_v2 "
            f"--batch_size 32 "
            # 추천하는 TRIDENT 명령어 옵션 조합
            f"--segmenter grandqc --remove_penmarks "
            f"--mag 20 --patch_size 256"
        )
        run_command(trident_cmd)

        # STEP 4: 청소 (원본 SVS 및 임시 UUID 폴더 삭제)
        print("🧹 Cleaning up...")
        # 원본 SVS 삭제
        shutil.rmtree(WSI_FLAT_DIR)
        os.makedirs(WSI_FLAT_DIR, exist_ok=True)
        # UUID 폴더 및 annotations.txt 삭제
        shutil.rmtree(WSI_TEMP_DIR)
        os.makedirs(WSI_TEMP_DIR, exist_ok=True)
        
        os.remove(batch_manifest_path)
        print(f"✅ Batch {i//BATCH_SIZE + 1} 완료!")

    except Exception as e:
        print(f"❌ Error in batch {i}: {e}")
        continue

print("\n🎉 All tasks finished.")