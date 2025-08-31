import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import librosa

# ---------------------------
# 1. 경로 설정
# ---------------------------
train_dir = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\train"
train_csv = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\REFERENCE2.csv"
model_dir = r"D:\MDO\heartbeat\1_New_HB_0818\1D_CNN\sliding_window\model"
os.makedirs(model_dir, exist_ok=True)

# ---------------------------
# 2. Dataset 정의 (CSV 기반 라벨 + Sliding Window + 패딩)
# ---------------------------
class HeartbeatDataset(Dataset):
    def __init__(self, data_dir, label_csv, window_size=16000, stride=8000):
        self.data = []
        self.labels = []

        # CSV 로딩
        df = pd.read_csv(label_csv)  # columns: filename, label
        file_to_label = dict(zip(df['filename'], df['label']))

        files = sorted(os.listdir(data_dir))
        for f in files:
            if f.endswith(".wav"):
                name = os.path.splitext(f)[0]  # a0001.wav → a0001
                if name in file_to_label:
                    path = os.path.join(data_dir, f)
                    sig, sr = librosa.load(path, sr=None)
                    label = int(file_to_label[name])



                # 짧은 신호 패딩
                if len(sig) < window_size:
                    sig = np.pad(sig, (0, window_size - len(sig)), mode='constant')
                    self.data.append(sig)
                    self.labels.append(label)
                else:
                    # sliding window
                    for start in range(0, len(sig) - window_size + 1, stride):
                        segment = sig[start:start + window_size]
                        self.data.append(segment)
                        self.labels.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][np.newaxis, :]  # (1, window_size)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

# ---------------------------
# 3. Conv1D 모델 정의
# ---------------------------
class Conv1DModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---------------------------
# 4. 데이터 로더 준비 (train/validation split)
# ---------------------------
full_dataset = HeartbeatDataset(train_dir, train_csv)

# train/validation 8:2 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------------------
# 5. 학습 설정
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv1DModel(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 6. 학습 루프 + Validation 평가
# ---------------------------
num_epochs = 15
for epoch in range(num_epochs):
    # --- 학습 ---
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # --- Validation 평가 ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

# ---------------------------
# 7. 모델 저장
# ---------------------------
model_path = os.path.join(model_dir, "heartbeat_conv1d.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")
