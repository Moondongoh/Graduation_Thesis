# # import torch
# # import pandas as pd
# # from torch.utils.data import TensorDataset, DataLoader
# # import torch.nn as nn

# # # CSV 불러오기
# # file_path = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_train.csv'
# # df = pd.read_csv(file_path)

# # X = df.drop('label', axis=1).values
# # y = df['label'].values
# # input_dim = X.shape[1]

# # X_tensor = torch.tensor(X, dtype=torch.float32)
# # y_tensor = torch.tensor(y, dtype=torch.long)
# # dataset = TensorDataset(X_tensor, y_tensor)
# # loader = DataLoader(dataset, batch_size=32, shuffle=False)

# # # CNN1D 모델 정의 (학습 때와 동일)
# # class CNN1D(nn.Module):
# #     def __init__(self, input_dim):
# #         super(CNN1D, self).__init__()
# #         self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
# #         self.pool = nn.MaxPool1d(2)
# #         self.fc1 = nn.Linear((input_dim // 2) * 64, 128)
# #         self.fc2 = nn.Linear(128, 2)

# #     def forward(self, x):
# #         x = x.unsqueeze(1)  # (B,1,L)
# #         x = torch.relu(self.conv1(x))
# #         x = self.pool(torch.relu(self.conv2(x)))
# #         x = x.view(x.size(0), -1)
# #         x = torch.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# # model = CNN1D(input_dim)
# # save_path = r'D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\cnn1d_wavelet_model.pth'
# # model.load_state_dict(torch.load(save_path, map_location="cpu"))
# # model.eval()

# # # 추론
# # correct, total = 0, 0
# # with torch.no_grad():
# #     for X_batch, y_batch in loader:
# #         outputs = model(X_batch)
# #         _, predicted = torch.max(outputs, 1)
# #         total += y_batch.size(0)
# #         correct += (predicted == y_batch).sum().item()

# # print(f"✅ CNN1D Test Accuracy: {100 * correct / total:.2f}%")


# import torch
# import pandas as pd
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn as nn

# # CSV 불러오기
# file_path = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_train.csv'
# df = pd.read_csv(file_path)

# X = df.drop('label', axis=1).values
# y = df['label'].values
# input_dim = X.shape[1]

# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.long)
# dataset = TensorDataset(X_tensor, y_tensor)
# loader = DataLoader(dataset, batch_size=32, shuffle=False)

# # CNN1D 모델 정의
# class CNN1D(nn.Module):
#     def __init__(self, input_dim):
#         super(CNN1D, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
#         self.fc1 = nn.Linear((input_dim // 2) * 64, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = x.unsqueeze(1)  # (B,1,L)
#         x = torch.relu(self.conv1(x))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 모델 불러오기
# model = CNN1D(input_dim)
# save_path = r'D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\cnn1d_wavelet_model.pth'
# model.load_state_dict(torch.load(save_path, map_location="cpu"))
# model.eval()

# # 추론 및 라벨 비교
# results = []
# with torch.no_grad():
#     for X_batch, y_batch in loader:
#         outputs = model(X_batch)
#         _, predicted = torch.max(outputs, 1)
#         for true_label, pred_label in zip(y_batch.tolist(), predicted.tolist()):
#             results.append((true_label, pred_label))

# # 결과 확인 (예: 처음 20개)
# for i, (true, pred) in enumerate(results[:10000]):
#     print(f"샘플 {i+1}: 실제 라벨 = {true}, 예측 라벨 = {pred}")

# # 전체 정확도
# correct = sum([true == pred for true, pred in results])
# total = len(results)
# print(f"✅ CNN1D Test Accuracy: {100 * correct / total:.2f}%")

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

file_path = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_train.csv"
df = pd.read_csv(file_path)

X = df.drop("label", axis=1).values
y = df["label"].values
input_dim = X.shape[1]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


# CNN1D 모델 정의
class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear((input_dim // 2) * 64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN1D(input_dim)
save_path = r"D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\cnn1d_wavelet_model.pth"
model.load_state_dict(torch.load(save_path, map_location="cpu"))
model.eval()

results = []
with torch.no_grad():
    for X_batch, y_batch in loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        for true_label, pred_label in zip(y_batch.tolist(), predicted.tolist()):
            results.append((true_label, pred_label))

# TP, TN, FP, FN 계산
TP = sum([1 for true, pred in results if true == 1 and pred == 1])
TN = sum([1 for true, pred in results if true == 0 and pred == 0])
FP = sum([1 for true, pred in results if true == 0 and pred == 1])
FN = sum([1 for true, pred in results if true == 1 and pred == 0])

# ***************민감도(확인 데이터) 및 특이도 계산***************
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

correct = sum([true == pred for true, pred in results])
total = len(results)
accuracy = 100 * correct / total

print(f"CNN1D Test Accuracy: {accuracy:.2f}%")
print(f"Sensitivity (민감도): {100 * sensitivity:.2f}%")
print(f"Specificity (특이도): {100 * specificity:.2f}%")
