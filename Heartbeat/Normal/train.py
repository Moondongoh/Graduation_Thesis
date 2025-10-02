import os
import random
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====================================================================
# 1. 설정 (Configuration)
# ====================================================================
# -- 입력/출력 경로 ---
TRAIN_FEATURES_CSV = r"D:\MDO\heartbeat\beat\Dataset\Wavelet\wavelet_features_train.csv"
MODEL_SAVE_PATH = r"D:\MDO\heartbeat\beat\Normal\model\Normal_model60.pt"
SCALER_SAVE_PATH = r"D:\MDO\heartbeat\beat\Normal\model\baseline_scaler60.pkl"

# -- 하이퍼파라미터 ---
SEED = 42
EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# 2. 재현성 고정 함수 *난수 고정
# ====================================================================
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================================================================
# 3. Conv1D 모델 정의 (모든 코드 통일 해야함)
# ====================================================================
class ConvFeatNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            # Layer 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Layer 3 (추가된 CNN 층)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Dropout(0.4), # 과적합 방지
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ====================================================================
# 4. 메인 학습 로직 (이전과 동일)
# ====================================================================
def main():
    set_seed(SEED)
    print(f"학습을 시작합니다. Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}, Validation Split: {VALIDATION_SPLIT}")

    # --- 데이터 준비 ---
    df = pd.read_csv(TRAIN_FEATURES_CSV)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    # 1. 훈련 / 검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=y
    )

    # 2. 스케일러 학습 및 적용
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. PyTorch DataLoader 생성
    train_dataset = TensorDataset(torch.tensor(X_train_scaled).unsqueeze(1), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled).unsqueeze(1), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

    # --- 모델, 손실함수, 옵티마이저 정의 ---
    model = ConvFeatNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 학습 및 검증 루프 ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_loss /= train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # --- 모델 및 스케일러 저장 ---
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n모델 저장 완료: {MODEL_SAVE_PATH}")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"스케일러 저장 완료: {SCALER_SAVE_PATH}")


if __name__ == "__main__":
    main()