# ============================================================
# 1. 생성된 학습 모델을 이용해 Validation 데이터 추론

# - 커스텀 Dataset 클래스 정의
# → validation 폴더에서 .png 이미지와 REFERENCE_binary.csv의 라벨을 매칭

# - 이미지 전처리 설정
# → Resize(224x224) 후 Tensor 변환

# - CSV 라벨 불러오기
# → REFERENCE_binary.csv를 딕셔너리로 변환해 Dataset에 전달

# -ResNet18 모델 불러오기
# → 클래스 수 2개에 맞게 FC 레이어 수정 후 학습된 모델 불러옴

# - 검증 이미지 추론 수행
# → 예측 결과 저장 및 정확도 출력

# - Confusion Matrix 출력
# → 시각화하여 성능 확인
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# 1. 커스텀 Dataset 클래스 정의
class HeartSoundValDataset(Dataset):
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

        return image, label, img_name


# 2. Transform 정의 (학습과 동일하게)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# 3. 라벨 로딩
label_df = pd.read_csv(
    r"E:/heartbeat/mels_images\validation/REFERENCE_binary.csv", header=None
)
label_df.columns = ["file_name", "label"]
label_dict = dict(zip(label_df["file_name"], label_df["label"]))

# 4. Dataset & DataLoader 생성
val_dataset = HeartSoundValDataset(
    image_dir="E:/heartbeat/mels_images/validation",
    label_dict=label_dict,
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(
    torch.load("../model/best_heart_cnn_model.pth", map_location=device)
)
model = model.to(device)
model.eval()

# 6. 추론 및 결과 저장
correct = 0
total = 0
results = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, img_names in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 저장용
        for img_name, pred in zip(img_names, preds):
            results.append((img_name, pred.item()))

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 7. 정확도 출력
accuracy = correct / total
print(f"✅ Validation Accuracy: {accuracy:.4f}")

# 8. 예측 결과 저장
df = pd.DataFrame(results, columns=["file_name", "predicted_label"])
df.to_csv("val_predictions.csv", index=False)
print("📁 예측 결과가 val_predictions.csv로 저장되었습니다.")

# 9. Confusion Matrix 시각화
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
