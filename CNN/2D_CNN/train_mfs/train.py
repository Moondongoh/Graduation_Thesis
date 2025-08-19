import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. PyTorch를 위한 커스텀 데이터셋 클래스
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

# 2. 데이터 로드 및 전처리 함수 (CSV 기반으로 수정)
def load_data_from_csv(data_path, csv_path):
    """
    CSV 파일을 기반으로 .npy 파일과 라벨을 로드하고 전처리합니다.
    """
    print(f"'{csv_path}' 파일에서 라벨 정보를 로딩합니다...")
    try:
        # CSV 파일에 헤더(제목 줄)가 없다고 가정하고 첫 번째 줄부터 데이터로 읽음
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다.")
        return None, None, None, None

    labels = []
    features = []
    max_len = 0

    print(f"'{data_path}' 디렉토리에서 .npy 파일 로딩을 시작합니다...")
    for index, row in df.iterrows():
        # 열 이름 대신 인덱스로 접근 (0: 파일명, 1: 라벨)
        filename_from_csv = str(row[0])
        label_from_csv = row[1]
        
        # .wav나 다른 확장자가 있다면 .npy로 변경합니다.
        base_filename = os.path.splitext(filename_from_csv)[0]
        npy_filename = f"{base_filename}.npy"
        file_path = os.path.join(data_path, npy_filename)
        
        if os.path.exists(file_path):
            feature = np.load(file_path)
            features.append(feature)
            labels.append(label_from_csv)
            
            if feature.shape[1] > max_len:
                max_len = feature.shape[1]
        else:
            print(f"경고: '{npy_filename}' 파일을 찾을 수 없습니다. 건너뜁니다.")

    if not features:
        print("오류: 로드할 수 있는 .npy 파일이 없습니다. 경로와 파일 이름을 확인해주세요.")
        return None, None, None, None
        
    print(f"데이터 로딩 완료. 총 {len(features)}개의 파일을 로드했습니다.")
    print(f"데이터의 최대 길이(시간 축): {max_len}")

    padded_features = []
    for feature in features:
        pad_width = max_len - feature.shape[1]
        padded_feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
        padded_features.append(padded_feature[np.newaxis, ...])

    X = np.array(padded_features)
    y = np.array(labels)

    # LabelEncoder를 사용하여 라벨을 0, 1, ... 정수로 변환
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("데이터 전처리 완료.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    return train_dataset, test_dataset, label_encoder.classes_, max_len

# 3. PyTorch CNN 모델 정의 (이전과 동일)
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=(3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = None

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        if self.fc_layers is None:
            num_features = x.shape[1]
            self.fc_layers = nn.Sequential(
                nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, self.num_classes)
            ).to(x.device)
        x = self.fc_layers(x)
        return x

# 4. 메인 실행 블록
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # --- 경로 설정 ---
    NPY_DATA_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv\processed_data'
    CSV_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv\processed_data\REFERENCE2.csv' # 실제 CSV 파일 경로
    MODEL_SAVE_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\binary_audio_cnn_model.pth' # 새로 저장될 모델 이름
    # -----------------
    
    train_dataset, test_dataset, class_names, max_len = load_data_from_csv(NPY_DATA_PATH, CSV_PATH)
    
    if train_dataset:
        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        num_classes = len(class_names)
        print(f"\n감지된 클래스: {class_names} (총 {num_classes}개)")
        model = AudioCNN(num_classes).to(device)

        try:
            data_iter = iter(train_loader)
            features, _ = next(data_iter)
            model(features.to(device))
            print("\n--- 모델 구조 ---")
            print(model)
            print("---------------------\n")
        except StopIteration:
            print("오류: 훈련 데이터로더가 비어있습니다.")
            exit()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        EPOCHS = 30
        print("모델 훈련을 시작합니다...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * features.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

        print("모델 훈련이 완료되었습니다.")

        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'max_len': max_len
        }, MODEL_SAVE_PATH)
        print(f"\n모델과 설정값을 '{MODEL_SAVE_PATH}'에 저장했습니다.")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total > 0:
            accuracy = 100 * correct / total
            print(f"테스트 정확도 (Accuracy): {accuracy:.2f}%")
