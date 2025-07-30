# -------------------------
# 1. 라이브러리 로드
# -------------------------
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# -------------------------
# 2. 라벨 로딩 및 처리
# -------------------------
label_df = pd.read_csv("E:/heartbeat/Dataset/REFERENCE.csv", header=None)
label_df.columns = ["file_name", "label"]
label_dict = dict(zip(label_df["file_name"], label_df["label"]))


# -------------------------
# 3. 커스텀 Dataset 정의
# -------------------------
class HeartSoundDataset(Dataset):
    def __init__(self, image_dir, label_dict, transform=None):
        self.image_dir = image_dir
        self.label_dict = label_dict
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if f.endswith(".png") and f.replace(".png", "") in label_dict
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.label_dict[img_name.replace(".png", "")]

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------
# 4. Transform 및 Dataset 분리, DataLoader 생성
# -------------------------
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = HeartSoundDataset(
    image_dir="E:/heartbeat/mels_images", label_dict=label_dict, transform=transform
)

# 학습/검증 데이터 분리 (80% / 20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------------
# 5. 모델 정의 (ResNet18)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# -------------------------
# 6. 검증 함수 정의
# -------------------------
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.long())

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc


# -------------------------
# 7. 학습 루프 (검증 포함)
# -------------------------
num_epochs = 100
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(
        f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # 검증 정확도 기준 최고 성능 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_heart_cnn_model.pth")
        print(f"✅ 새로운 최고 검증 정확도 {best_val_acc:.4f}로 모델 저장 완료")
