import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# ====================================================================
# --- 평가할 모델 선택 ---
# ====================================================================

## 4. GAN 모델
MODEL_PATH = r"D:\MDO\heartbeat\beat\Gan\model\gan_model60.pt"
SCALER_PATH = r"D:\MDO\heartbeat\beat\Gan\model\gan_scaler60.pkl"

# --- 테스트 데이터 경로 ---
TEST_CSV_PATH = r"D:\MDO\heartbeat\beat\Dataset\Wavelet\wavelet_features_test.csv"

# ====================================================================
# GPU사용 여부 췌크~
# ====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# Conv1D 모델 정의 (학습 때 사용한 것과 반드시 동일해야 함)
# ====================================================================
class ConvFeatNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.AdaptiveAvgPool1d(1),
            
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ====================================================================
# 메인 평가 로직
# ====================================================================
def evaluate():
    """지정된 모델과 스케일러를 사용하여 테스트 데이터셋의 성능을 평가합니다."""
    
    print("="*50)
    print(f"모델 평가 시작: {Path(MODEL_PATH).stem}")
    print(f"테스트 데이터: {TEST_CSV_PATH}")
    print("="*50)

    # --- 데이터 로드 및 전처리 ---
    df_test = pd.read_csv(TEST_CSV_PATH)
    X_test = df_test.drop('label', axis=1).values.astype(np.float32)
    y_test = df_test['label'].values.astype(np.int64)

    try:
        scaler = joblib.load(SCALER_PATH)
        X_test_scaled = scaler.transform(X_test)
    except FileNotFoundError:
        print(f"⚠️ 경고: 스케일러 파일을 찾을 수 없습니다 ({SCALER_PATH}). 스케일링 없이 진행합니다.")
        X_test_scaled = X_test
        
    test_dataset = TensorDataset(torch.tensor(X_test_scaled).unsqueeze(1), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --- 모델 로드 ---
    model = ConvFeatNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ 에러: 모델 파일을 찾을 수 없습니다 ({MODEL_PATH}). 평가를 중단합니다.")
        return
        
    model.eval()

    # --- 추론 ---
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(DEVICE)
            outputs = model(features)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # --- 성능 지표 계산 및 출력 ---
    y_true = y_test
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    print("\n### 최종 성능 평가 결과 ###\n")
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = report['1']['precision']
    f1_score = report['1']['f1-score']

    print("--- 요약 ---")
    print(f"1. Accuracy (정확도)   : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"2. Sensitivity (민감도): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"3. Specificity (특이도) : {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"4. Precision (정밀도)   : {precision:.4f} ({precision*100:.2f}%)")
    print(f"5. F1-Score (F1 점수)   : {f1_score:.4f}")
    print(f"6. AUC (ROC AUC)      : {auc:.4f}")
    print("\n")


if __name__ == "__main__":
    evaluate()