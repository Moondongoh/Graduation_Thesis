import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 불러오기
file_path = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_train.csv'
try:
    df = pd.read_csv(file_path)
    print(f"✅ 데이터 로드 완료: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다. 라벨 추가 코드를 먼저 실행해주세요.")

# 2. 특징(X)과 라벨(y) 분리
X = df.drop('label', axis=1)
y = df['label']

# 3. 데이터셋 분할 및 SMOTE 적용
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\n✅ SMOTE 적용 완료. 학습 데이터 라벨 분포: \n{y_train_resampled.value_counts()}")

# 4. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)
print("\n✅ 모델 학습 완료")

# 5. 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- 모델 성능 평가 (원본 테스트 데이터 기준) ---")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"균형 정확도 (Balanced Accuracy): {bal_acc:.4f}")
print("\n분류 리포트:")
print(report)

# 6. Feature Importance 분석
importances = model.feature_importances_
indices = np.argsort(importances)[-15:][::-1]  # 상위 15개

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices][::-1], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 15 중요한 웨이블릿 특징")
plt.tight_layout()
plt.show()

# 7. 모델 저장
model_file_path = r'D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\random_forest_model.joblib'
joblib.dump(model, model_file_path)
print(f"\n🎉 모델이 '{model_file_path}'에 저장되었습니다.")
