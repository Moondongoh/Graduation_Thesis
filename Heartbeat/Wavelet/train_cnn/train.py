import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# 1. CSV 불러오기
file_path = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_train.csv'
df = pd.read_csv(file_path)

X = df.drop('label', axis=1).values
y = df['label'].values
input_dim = X.shape[1]

# 2. Tensor 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

# Train/Test 분리
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. CNN1D 모델 정의
class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear((input_dim // 2) * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # 이진 분류

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,L)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))  # (B,64,L/2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN1D(input_dim).to("cpu")

# 4. 학습 준비
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
for epoch in range(15):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 6. 평가
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

# 7. 모델 저장
save_path = r'D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\cnn1d_wavelet_model.pth'
torch.save(model.state_dict(), save_path)
print(f"✅ 모델이 저장되었습니다: {save_path}")
