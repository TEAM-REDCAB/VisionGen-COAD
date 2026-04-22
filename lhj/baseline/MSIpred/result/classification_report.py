import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 예시 데이터 생성 (사용자님의 실제 데이터 변수명으로 바꾸시면 됩니다)
# df_true: 실제 정답 데이터, df_pred: MSIpred로 예측한 데이터
actual = pd.read_csv("/home/team1/lhj/VisionGen-COAD/lhj/raw/common_patients.txt", sep="\t")
actual = actual['type']
predicted = pd.read_csv("/home/team1/lhj/VisionGen-COAD/lhj/baseline/MSIpred/result/MSIpred_result.tsv", sep="\t")
predicted = predicted['type']


# 2. 결과 저장 경로 설정
output_dir = "/home/team1/lhj/VisionGen-COAD/lhj/baseline/MSIpred/result/"
txt_filename = output_dir + "MSIpred_performance_metrics.txt"
img_filename = output_dir + "MSIpred_confusion_matrix.png"

# 3. 분석 수행
labels = ['MSS', 'MSIMUT']
cm = confusion_matrix(actual, predicted, labels=labels)
cm_df = pd.DataFrame(cm, index=['Actual_MSS', 'Actual_MSIMUT'], 
                         columns=['Pred_MSS', 'Pred_MSIMUT'])
report = classification_report(actual, predicted, target_names=labels)
acc = accuracy_score(actual, predicted)

# 4. 텍스트 지표 파일로 저장
with open(txt_filename, "w") as f:
    f.write("### MSIpred Classification Performance ###\n\n")
    f.write("--- Confusion Matrix ---\n")
    f.write(cm_df.to_string())
    f.write("\n\n--- Classification Report ---\n")
    f.write(report)
    f.write("\nOverall Accuracy: {:.2f}%\n".format(acc * 100))

print("수치 결과가 저장되었습니다: {}".format(txt_filename))

# 5. Confusion Matrix 이미지 파일로 저장
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('MSI Prediction Confusion Matrix')

# plt.show() 대신 저장
plt.savefig(img_filename, dpi=300, bbox_inches='tight')
plt.close() # 메모리 해제

print("이미지 결과가 저장되었습니다: {}".format(img_filename))