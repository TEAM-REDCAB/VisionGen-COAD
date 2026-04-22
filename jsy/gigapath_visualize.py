import os
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader

import config as cf
from h5dataset import H5Dataset

def visualize_pca_side_by_side():
    output_dir = os.path.join(cf.get_results_path(), 'visualizations_pca_compare')
    os.makedirs(output_dir, exist_ok=True)
    
    # 썸네일 폴더 경로 (실제 경로로 맞춰주세요)
    THUMBNAIL_DIR = './working/trident_processed/thumbnails/' 
    
    for fold in range(5):
        print(f"\n🎨 PCA Side-by-Side Fold {fold}...")
        
        test_dataset = H5Dataset(split="test", fold_col=f'fold_{fold}')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        save_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)

        for idx, (features, coords, label) in enumerate(tqdm(test_loader, desc=f"Fold {fold}")):
            patient_id = test_dataset.df.iloc[idx]['patient']
            true_label_val = int(label.item())
            label_str = "MSI" if true_label_val == 1 else "MSS"
            
            patch_features = features.squeeze(0).cpu().numpy().astype(np.float32)
            coords_array = coords.squeeze(0).cpu().numpy()
            
            # --- 1. PCA 변환 ---
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(patch_features)
            
            for c in range(3):
                min_val = pca_features[:, c].min()
                max_val = pca_features[:, c].max()
                pca_features[:, c] = (pca_features[:, c] - min_val) / (max_val - min_val + 1e-8)

            # --- 2. 256px 고정 캔버스 생성 (조직 형태 완벽 보존) ---
            patch_size = 256
            x_coords = coords_array[:, 0]
            y_coords = coords_array[:, 1]
            
            grid_x = np.round((x_coords - x_coords.min()) / patch_size).astype(int)
            grid_y = np.round((y_coords - y_coords.min()) / patch_size).astype(int)
            
            width = grid_x.max() + 1
            height = grid_y.max() + 1
            
            # 하얀색 배경 캔버스
            pca_canvas = np.ones((height, width, 3), dtype=np.float32)
            pca_canvas[grid_y, grid_x] = pca_features

            # --- 3. 썸네일 원본 로드 ---
            original_img = None
            if os.path.exists(THUMBNAIL_DIR):
                all_thumbnails = os.listdir(THUMBNAIL_DIR)
                matching_thumbnails = [f for f in all_thumbnails if f.startswith(patient_id)]
                
                if matching_thumbnails:
                    thumbnail_path = os.path.join(THUMBNAIL_DIR, matching_thumbnails[0])
                    original_img = cv2.imread(thumbnail_path)
                    if original_img is not None:
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # --- 4. 나란히(Side-by-Side) 렌더링 ---
            # 원본 이미지가 없으면 PCA 단독으로, 있으면 나란히 플롯
            if original_img is not None:
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                axes[0].imshow(original_img)
                axes[0].set_title("Original WSI Thumbnail", fontsize=14)
                axes[0].axis('off')
                
                axes[1].imshow(pca_canvas, interpolation='none')
                axes[1].set_title(f"GigaPath PCA Feature Map (True: {label_str})", fontsize=14)
                axes[1].axis('off')
                
                fig.suptitle(f"Patient ID: {patient_id}", fontsize=18, fontweight='bold')
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(pca_canvas, interpolation='none')
                ax.set_title(f"PCA Feature Map | {patient_id} | True: {label_str}")
                ax.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"Compare_PCA_{label_str}_{patient_id}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            gc.collect()

if __name__ == "__main__":
    visualize_pca_side_by_side()