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
MODEL_SAVE_PATH = r"D:\MDO\heartbeat\beat\Gan\model\gan_model60.pt"
SCALER_SAVE_PATH = r"D:\MDO\heartbeat\beat\Gan\model\gan_scaler60.pkl"

# -- 분류 모델 하이퍼파라미터 ---
SEED = 42
EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- GAN 하이퍼파라미터 ---
FEATURE_DIM = 36  # 36차원 특징
LATENT_DIM = 100  # 생성자 입력 노이즈 차원
GAN_EPOCHS = 5000 # GAN 학습 횟수
GAN_LR = 2e-4
GAN_BATCH_SIZE = 64

# ====================================================================
# 2. 재현성 고정 함수
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
# 3. 모델 정의
# ====================================================================

# --- GAN 모델 ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, FEATURE_DIM) # 최종 출력: 36차원 특징 벡터
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256), # 입력: 36차원 특징 벡터
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),
            
            nn.Sigmoid() # 진짜일 확률 (0~1)
        )
    def forward(self, x):
        return self.model(x)

# --- Conv1D 분류 모델 ---
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
# 4. 메인 학습 로직
# ====================================================================
def main():
    set_seed(SEED)
    print(f"개선된 GAN 모델 학습을 시작합니다. Device: {DEVICE}")

    # --- 데이터 준비 ---
    df = pd.read_csv(TRAIN_FEATURES_CSV)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    # 1. 훈련 / 검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y
    )

    # 2. 스케일러 학습 및 적용
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- GAN 학습 단계 ---
    print("\n--- 1단계: GAN 학습 시작 ---")
    # 3. GAN 학습을 위해 훈련 데이터에서 소수 클래스(라벨 1)만 추출
    minority_features = torch.tensor(X_train_scaled[y_train == 1], dtype=torch.float32).to(DEVICE)
    minority_loader = DataLoader(TensorDataset(minority_features), batch_size=GAN_BATCH_SIZE, shuffle=True)
    
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    g_optimizer = optim.Adam(generator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=GAN_LR, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    for epoch in range(1, GAN_EPOCHS + 1):
        for i, (real_samples,) in enumerate(minority_loader):
            real = real_samples.to(DEVICE)
            
            # 판별자 학습
            d_optimizer.zero_grad()
            z = torch.randn(real.size(0), LATENT_DIM, device=DEVICE)
            fake = generator(z)
            
            real_loss = adversarial_loss(discriminator(real), torch.ones(real.size(0), 1, device=DEVICE))
            fake_loss = adversarial_loss(discriminator(fake.detach()), torch.zeros(real.size(0), 1, device=DEVICE))
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            d_optimizer.step()

            # 생성자 학습
            g_optimizer.zero_grad()
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), torch.ones(real.size(0), 1, device=DEVICE))
            
            g_loss.backward()
            g_optimizer.step()

        if epoch % (GAN_EPOCHS // 10) == 0:
            print(f"GAN Epoch {epoch}/{GAN_EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    
    print("--- GAN 학습 완료 ---")

    # --- 데이터 합성 및 분류기 학습 준비 ---
    print("\n--- 2단계: 데이터 합성 및 분류기 학습 준비 ---")
    num_majority = np.sum(y_train == 0)
    num_minority = np.sum(y_train == 1)
    num_to_generate = num_majority - num_minority

    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_to_generate, LATENT_DIM, device=DEVICE)
        generated_features = generator(z).cpu().numpy()

    X_train_resampled = np.vstack((X_train_scaled, generated_features))
    y_train_resampled = np.hstack((y_train, np.ones(num_to_generate, dtype=np.int64)))
    
    print(f"GAN 적용 전 훈련 데이터 shape: {X_train_scaled.shape}")
    print(f"GAN 적용 후 훈련 데이터 shape: {X_train_resampled.shape}")
    print(f"검증 데이터 shape: {X_val_scaled.shape}")
    print(f"GAN 적용 후 라벨 분포: {np.bincount(y_train_resampled)}")

    # DataLoader 생성
    train_dataset = TensorDataset(torch.tensor(X_train_resampled).unsqueeze(1), torch.tensor(y_train_resampled))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled).unsqueeze(1), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 분류기 학습 ---
    print("\n--- 3단계: Conv1D 분류기 학습 시작 ---")
    classifier_model = ConvFeatNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        classifier_model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = classifier_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
            train_total += labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_loss /= train_total

        classifier_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = classifier_model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                val_total += labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= val_total
        print(f"Classifier Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")

    # --- 모델 및 스케일러 저장 ---
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n분류기 모델 저장 완료: {MODEL_SAVE_PATH}")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"스케일러 저장 완료: {SCALER_SAVE_PATH}")


if __name__ == "__main__":
    main()