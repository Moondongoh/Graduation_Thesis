import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob

# -------------------
# 1. Dataset 정의
# -------------------
class WavDataset(Dataset):
    def __init__(self, folder, sr=16000):
        self.files = glob(os.path.join(folder, "*.wav"))
        self.sr = sr

        # 모든 파일 길이 확인 → max_len 결정
        lengths = []
        for f in self.files:
            y, _ = librosa.load(f, sr=self.sr)
            lengths.append(len(y))
        self.max_len = max(lengths)
        print(f"📏 Train set max length: {self.max_len} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        y, _ = librosa.load(filepath, sr=self.sr)

        # 패딩
        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)), mode="constant")
        else:
            y = y[:self.max_len]

        x = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        # 라벨링 예시: 파일명 기준
        label = 0 if "normal" in filepath else 1
        return x, label

# -------------------
# 2. 1D CNN 모델 정의
# -------------------
class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=2, max_len=160000):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)

        # 두 번 pooling stride=4 → 16
        conv_out_len = max_len // 16
        self.fc1 = nn.Linear(32 * conv_out_len, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------
# 3. 학습 루프
# -------------------
if __name__ == "__main__":
    train_folder = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\train"
    dataset = WavDataset(train_folder)
    max_len = dataset.max_len
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Simple1DCNN(num_classes=2, max_len=max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

    # -------------------
    # 4. 모델 + max_len 저장
    # -------------------
    save_path = r"D:\MDO\heartbeat\1_New_HB_0818\1D_CNN\model\1d_cnn_model.pth"
    torch.save({
        'state_dict': model.state_dict(),
        'max_len': max_len
    }, save_path)
    print(f"✅ 모델과 max_len이 {save_path} 에 저장되었습니다.")
