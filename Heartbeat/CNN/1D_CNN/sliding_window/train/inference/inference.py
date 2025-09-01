import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
from collections import defaultdict

# ---------------------------
# 1. 경로 설정
# ---------------------------
val_dir = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\validation"
val_csv = os.path.join(val_dir, "REFERENCE2.csv")  # validation 라벨 CSV
model_path = r"D:\MDO\heartbeat\1_New_HB_0818\1D_CNN\sliding_window\model\heartbeat_conv1d.pth"
window_size = 16000
stride = 8000
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print("Loading validation dataset...")

# ---------------------------
# 2. Dataset 정의 (Test용)
# ---------------------------
class ValDataset(Dataset):
    def __init__(self, data_dir, label_csv, window_size=16000, stride=8000):
        self.data = []
        self.labels = []
        self.file_map = []

        # CSV 읽기 (헤더 없는 경우)
        df = pd.read_csv(label_csv, names=['filename', 'label'])
        file_to_label = dict(zip(df['filename'], df['label']))

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav")])
        for f in files:
            name = os.path.splitext(f)[0]
            if name in file_to_label:
                path = os.path.join(data_dir, f)
                sig, sr = librosa.load(path, sr=None)
                label = int(file_to_label[name])

                # 짧은 신호 패딩
                if len(sig) < window_size:
                    sig = np.pad(sig, (0, window_size - len(sig)), mode='constant')
                    self.data.append(sig)
                    self.labels.append(label)
                    self.file_map.append(f)
                else:
                    for start in range(0, len(sig) - window_size + 1, stride):
                        segment = sig[start:start + window_size]
                        self.data.append(segment)
                        self.labels.append(label)
                        self.file_map.append(f)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Total segments: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][np.newaxis, :]
        y = self.labels[idx]
        f = self.file_map[idx]
        return torch.tensor(x), y, f

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
# 4. 데이터 로더
# ---------------------------
val_dataset = ValDataset(val_dir, val_csv, window_size, stride)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Number of validation segments: {len(val_dataset)}")

# ---------------------------
# 5. 모델 로드
# ---------------------------
model = Conv1DModel(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded.")

# ---------------------------
# 6. 추론 및 파일 단위 투표
# ---------------------------
pred_dict = defaultdict(list)
label_dict = {}  # 파일별 실제 라벨

with torch.no_grad():
    for i, (x, y, files) in enumerate(val_loader):
        x = x.to(device)
        out = model(x)
        pred = out.argmax(dim=1).cpu().numpy()
        y = y.numpy()
        for f, p, t in zip(files, pred, y):
            pred_dict[f].append(p)
            label_dict[f] = t
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} batches")

# ---------------------------
# 7. 파일별 최종 예측
# ---------------------------
final_preds = {}
for f, preds in pred_dict.items():
    final_preds[f] = int(np.bincount(preds).argmax())

# ---------------------------
# 8. 정확도 계산
# ---------------------------
correct = 0
total = len(final_preds)
for f, pred in final_preds.items():
    if pred == label_dict[f]:
        correct += 1

val_acc = correct / total * 100
print(f"Validation Accuracy: {val_acc:.2f}%")
