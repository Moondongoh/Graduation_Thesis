import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd

# -------------------
# 1. Test Dataset 정의
# -------------------
class WavTestDataset(Dataset):
    def __init__(self, folder, max_len, sr=16000):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]
        self.sr = sr
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        y, _ = librosa.load(filepath, sr=self.sr)

        if len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)), mode="constant")
        else:
            y = y[:self.max_len]

        x = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, os.path.basename(filepath)

# -------------------
# 2. 1D CNN 모델 정의
# -------------------
class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=2, max_len=160000):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
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
# 3. Main: test 추론
# -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 경로
    model_path = r"D:\MDO\heartbeat\1_New_HB_0818\1D_CNN\model\1d_cnn_model.pth"

    # 체크포인트 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    max_len = checkpoint['max_len']

    # 모델 정의 및 가중치 로드
    model = Simple1DCNN(num_classes=2, max_len=max_len).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Test 데이터셋 & DataLoader
    test_folder = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\validation"
    test_dataset = WavTestDataset(test_folder, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------
    # CSV 기반 정답 라벨 불러오기
    # -------------------
    csv_path = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\validation\REFERENCE2.csv"
    df_label = pd.read_csv(csv_path, header=None)
    df_label.columns = ['filename', 'label']

    # 파일명에 .wav 확장자 붙이기
    label_dict = {f"{fname}.wav": label for fname, label in zip(df_label['filename'], df_label['label'])}

    # -------------------
    # 추론 및 정확도 계산
    # -------------------
    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, fname in test_loader:
            x = x.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1).item()

            # CSV에서 실제 라벨 가져오기
            true_label = label_dict[fname[0]]  # fname[0] = 파일명

            if pred == true_label:
                correct += 1
            total += 1

            results.append((fname[0], pred, true_label))

    # 정확도 계산
    accuracy = correct / total * 100
    print(f"✅ Validation Accuracy: {accuracy:.2f}%")

    # CSV 저장 (pred + 실제 라벨)
    df = pd.DataFrame(results, columns=["filename", "pred_label", "true_label"])
    df.to_csv("validation_predictions.csv", index=False, encoding="utf-8-sig")
    print("✅ validation_predictions.csv 저장 완료")
