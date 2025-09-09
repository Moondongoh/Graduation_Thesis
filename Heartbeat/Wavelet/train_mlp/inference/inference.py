# import torch
# import pandas as pd
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# # CSV 불러오기
# file_path = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_validation.csv'
# df = pd.read_csv(file_path)

# X = df.drop('label', axis=1).values
# y = df['label'].values

# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.long)

# dataset = TensorDataset(X_tensor, y_tensor)
# loader = DataLoader(dataset, batch_size=32, shuffle=False)

# # MLP 모델 정의 (학습 때랑 동일)
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
#         self.fc3 = nn.Linear(hidden_dim//2, 2)  # 이진 분류

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# input_dim = X.shape[1]
# model = MLP(input_dim)
# # 모델 불러오기
# save_path = r'D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\mlp_wavelet_model.pth'
# model.load_state_dict(torch.load(save_path, map_location="cpu"))
# model.eval()

# # 추론
# correct, total = 0, 0
# with torch.no_grad():
#     for X_batch, y_batch in loader:
#         outputs = model(X_batch)
#         _, predicted = torch.max(outputs, 1)
#         total += y_batch.size(0)
#         correct += (predicted == y_batch).sum().item()

# print(f"✅ MLP Test Accuracy: {100 * correct / total:.2f}%")


import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

file_path = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_validation.csv"
df = pd.read_csv(file_path)

if "label" not in df.columns:
    print("❌ 'label' 열을 찾을 수 없습니다. 올바른 CSV 파일을 사용했는지 확인하세요.")
else:
    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # MLP 모델 정의
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, 2)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_dim = X.shape[1]
    model = MLP(input_dim)

    save_path = r"D:\MDO\heartbeat\1_New_HB_0818\Wavelet\model\mlp_wavelet_model.pth"
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

    print(f"MLP Test Accuracy: {accuracy:.2f}%")
    print(f"Sensitivity (민감도): {100 * sensitivity:.2f}%")
    print(f"Specificity (특이도): {100 * specificity:.2f}%")
