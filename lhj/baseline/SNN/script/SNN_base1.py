import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap
import os

# 1. 데이터 로드
matrix = np.load('/home/team1/lhj/VisionGen-COAD/lhj/preprocessing/genomic_input_matrix.npy') # (N, 1425, 9)
with open('/home/team1/lhj/VisionGen-COAD/lhj/preprocessing/genomic_encoding_states.pkl', 'rb') as f:
    states = pickle.load(f)

patient_list = states['patient_list']  # 어레이와 순서가 동일한 환자 명단
vocab_size = len(states['func_vocab']) + 1  # 패딩 0 포함한 기능 사전 크기

# 2. 정답 레이블링 (Labeling)
clinical_df = pd.read_csv('/home/team1/lhj/VisionGen-COAD/lhj/raw/common_patients.txt', sep='\t')
clinical_df['type'] = clinical_df['type'].map({'MSIMUT': 1, 'MSS': 0})
label_map = dict(zip(clinical_df['patient'], clinical_df['type']))

# 어레이 순서에 맞춰 레이블 리스트 생성
labels = [label_map[pid] for pid in patient_list]
y = np.array(labels)

# 3. 카운트 벡터화 (Multi-Hot Encoding) 함수
def get_patient_functional_summary(matrix, vocab_size):
    num_patients = matrix.shape[0]
    # 각 환자별로 (vocab_size) 크기의 빈 벡터 생성
    summary = np.zeros((num_patients, vocab_size))
    
    # 기능 채널 (인덱스 2~7) 추출
    func_data = matrix[:, :, 2:8].astype(int)
    
    for p in range(num_patients):
        # 해당 환자의 모든 변이에서 나타난 모든 기능 ID 추출 (0 제외)
        patient_funcs = func_data[p].flatten()
        unique_funcs = patient_funcs[patient_funcs > 0]
        
        # 각 기능 ID가 몇 번 등장했는지 카운트하여 빈도 벡터 생성
        for f_id in unique_funcs:
            summary[p, f_id] += 1
            
    return summary

def get_advanced_snn_features(matrix, states, vocab_size):
    num_patients = matrix.shape[0]
    
    # 1. Gene Function (2~7번 채널)
    X_func_raw = get_patient_functional_summary(matrix, vocab_size)

    # 2. Variant Classification (1번 채널)
    vc_vocab = states.get('vc_vocab', {})
    vc_vocab_size = len(vc_vocab) + 1
    X_vc_raw = np.zeros((num_patients, vc_vocab_size))
    
    # 3. Hugo_HGVSc (0번 채널)
    var_vocab = states.get('var_vocab', {})
    var_vocab_size = len(var_vocab) + 1
    X_var_raw = np.zeros((num_patients, var_vocab_size))

    # 4. t_vaf (8번 채널) - 히스토그램은 그대로 사용
    X_vaf_dist = np.zeros((num_patients, 10))

    for p in range(num_patients):
        # VC 카운트
        p_vc = matrix[p, :, 1].astype(int)
        for v_id in p_vc[p_vc > 0]:
            if v_id < vc_vocab_size: X_vc_raw[p, v_id] += 1
            
        # VarID 카운트
        p_var = matrix[p, :, 0].astype(int)
        for v_id in p_var[p_var > 0]:
            if v_id < var_vocab_size: X_var_raw[p, v_id] += 1
            
        # VAF 분포
        p_vaf = matrix[p, :, 8]
        p_vaf = p_vaf[p_vaf > 0]
        if len(p_vaf) > 0:
            counts, _ = np.histogram(p_vaf, bins=10, range=(0, 1))
            X_vaf_dist[p] = counts

    # [핵심] 여기서 0번 컬럼(Padding)을 제외하고 합칩니다.
    X_combined = np.concatenate([
        X_var_raw[:, 1:],  # 0번 제외
        X_vc_raw[:, 1:],   # 0번 제외
        X_func_raw[:, 1:], # 0번 제외
        X_vaf_dist         # VAF는 그대로
    ], axis=1)
    
    # 각 섹션의 '수정된' 차원 정보를 반환 (SHAP 이름 매칭용)
    dims = (X_var_raw.shape[1]-1, X_vc_raw.shape[1]-1, X_func_raw.shape[1]-1)
    
    return X_combined, dims

# 1. SNN 모델 정의
class MSI_SNN_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64): # 256 -> 64로 대폭 축소
        super(MSI_SNN_Baseline, self).__init__()
        
        self.network = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(p=0.5), # 0.1 -> 0.5로 강화 (절반을 끄고 학습)
            
            # Hidden Layer
            nn.Linear(hidden_dim, hidden_dim // 2), # 32 units
            nn.SELU(),
            nn.AlphaDropout(p=0.3),
            
            # Output Layer
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

# 최종 입력 데이터 결합 (N, vocab_size + 1)
X_snn, dims = get_advanced_snn_features(matrix, states, vocab_size)
var_dim, vc_dim, func_dim = dims

# 1. K-Fold 설정 (5-Fold)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_all = y  # 전체 레이블

fold_results = []
epochs = 100
total_metrics = []

final_model = None
final_X_train = None
final_X_test = None

all_fold_losses = []
all_fold_probs = []
all_fold_y_true = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_snn, y_all)):
    print(f"\n# Fold {fold+1} Training...")
    
    # 데이터 분할
    X_train_fold = torch.FloatTensor(X_snn[train_idx])
    y_train_fold = torch.FloatTensor(y_all[train_idx]).view(-1, 1)
    X_val_fold = torch.FloatTensor(X_snn[val_idx])
    y_val_fold = torch.FloatTensor(y_all[val_idx]).view(-1, 1)
    
    # [DataLoader 생성] - 매 Fold마다 새로 생성
    train_ds = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = MSI_SNN_Baseline(input_dim=X_snn.shape[1], hidden_dim=64)

    # 클래스 가중치 재계산 (Fold마다 분포가 다를 수 있음)
    n_mss = (y_train_fold == 0).sum().item()
    n_msi = (y_train_fold == 1).sum().item()
    pos_weight = torch.tensor([n_mss / n_msi])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 4. 옵티마이저 규제 강화 (weight_decay 1e-3 -> 1e-2)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)

    # 학습 루프 (Training Loop)
    train_losses = [] # Fold마다 리셋
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

    all_fold_losses.append(train_losses)
    # --- 검증 (Validation) ---
    model.eval()
    with torch.no_grad():
        logits = model(X_val_fold)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        y_true_fold = y_val_fold.numpy()
        
        all_fold_probs.append(probs)
        all_fold_y_true.append(y_true_fold)
        
        # 지표 계산
        auroc = roc_auc_score(y_true_fold, probs)
        recall = recall_score(y_true_fold, preds)
        precision = precision_score(y_true_fold, preds, zero_division=0)
        acc = accuracy_score(y_true_fold, preds)
        f1 = f1_score(y_true_fold, preds)
        
        total_metrics.append([auroc, recall, precision, acc, f1])
        print(f"Fold {fold+1} Result - AUROC: {auroc:.4f}, Recall: {recall:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")
    
    # SHAP 분석을 위해 마지막 폴드의 상태를 저장
    if fold == n_splits - 1:
        final_model = model
        final_X_train = X_train_fold
        final_X_test = X_val_fold
        final_probs = probs # ROC 커브용
        final_y_true = y_true_fold

# 5. 최종 평가 (Evaluation)
# 1. 지표 계산
avg_metrics = np.mean(total_metrics, axis=0)
metrics = {
    "AUROC": avg_metrics[0],
    "Precision": avg_metrics[2],
    "Recall": avg_metrics[1],
    "F1": avg_metrics[4],
    "Acc": avg_metrics[3]
}
save_dir = '/home/team1/lhj/VisionGen-COAD/lhj/baseline/SNN/result'
os.makedirs(save_dir, exist_ok=True)

results_path = os.path.join(save_dir, 'detailed_performance_metrics.txt')
with open(results_path, 'w') as f:
    f.write("="*40 + "\n")
    f.write("SNN Baseline Detailed Performance\n")
    f.write("-" * 40 + "\n")
    f.write(f"AUROC:       {metrics['AUROC']:.4f}\n")
    f.write(f"Precision:   {metrics['Precision']:.4f}\n")
    f.write(f"Recall:      {metrics['Recall']:.4f}\n")
    f.write(f"F1-Score:    {metrics['F1']:.4f}\n")
    f.write(f"Accuracy:    {metrics['Acc']:.4f}\n")
    f.write("="*40 + "\n")
print(f"성능 지표가 '{results_path}'에 저장되었습니다.")
    
#2. ROC 커브 시각화 및 저장
fpr, tpr, _ = roc_curve(final_y_true, final_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))

for i in range(n_splits):
    # 각 폴드별로 fpr, tpr 계산
    fpr, tpr, _ = roc_curve(all_fold_y_true[i], all_fold_probs[i])
    fold_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.6, label=f'Fold {i+1} (AUC = {fold_auc:.2f})')
    
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('SNN Baseline: 5-Fold ROC Curves')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

roc_path = os.path.join(save_dir, 'snn_kfold_roc_curves.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"ROC 커브가 '{roc_path}'에 저장되었습니다.")

# 3. SHAP 분석 
explainer = shap.DeepExplainer(final_model, final_X_train[:100]) 
shap_values = explainer.shap_values(final_X_test)

if isinstance(shap_values, list):
    shap_values = shap_values[0]
    
# 각 카테고리 구간 설정
var_end = var_dim
vc_end = var_dim + vc_dim
func_end = var_dim + vc_dim + func_dim

cat_imp = {
    'Hugo_HGVSc': np.abs(shap_values[:, 0:var_end]).mean(axis=0).sum(),
    'Variant_Classification': np.abs(shap_values[:, var_end:vc_end]).mean(axis=0).sum(),
    'Gene_Function': np.abs(shap_values[:, vc_end:func_end]).mean(axis=0).sum(),
    't_vaf': np.abs(shap_values[:, func_end:]).mean(axis=0).sum()
}

# 3. 시각화
plt.figure(figsize=(10, 6))
plt.barh(list(cat_imp.keys()), list(cat_imp.values()), color='salmon')
plt.title('Genomic Category Importance (Sum of SHAP)')
plt.xlabel('Global Importance')
plt.gca().invert_yaxis()
plt.savefig(os.path.join(save_dir, 'category_importance.png'), bbox_inches='tight')
    
# 4. 학습 곡선 시각화
avg_train_losses = np.mean(all_fold_losses, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(avg_train_losses, label='Average Train Loss', color='blue', lw=2)
plt.title('SNN 5-Fold Average Training Loss', fontsize=15)
plt.xlabel('Epochs')
plt.ylabel('Loss (BCE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(save_dir, 'snn_average_loss.png'))
plt.close()
print(f"학습 곡선 그래프가 '{os.path.join(save_dir, 'snn_average_loss.png')}'에 저장되었습니다.")

# 학습된 모델 가중치 저장
model_path = os.path.join(save_dir, 'snn_model_weights.pth')
torch.save(model.state_dict(), model_path)
print(f"모델 가중치가 '{model_path}'에 저장되었습니다.")

