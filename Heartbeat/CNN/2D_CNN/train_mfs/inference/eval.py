import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 1. 훈련 시 사용했던 모델과 동일한 구조의 클래스를 정의합니다.
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

# 2. 단일 파일을 예측하는 함수
def predict_single_file(model, file_path, max_len, device):
    feature = np.load(file_path)
    pad_width = max_len - feature.shape[1]
    if pad_width < 0:
        feature = feature[:, :max_len]
        pad_width = 0
    padded_feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
    tensor_feature = torch.tensor(padded_feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_feature)
        _, predicted_idx = torch.max(output.data, 1)
    return predicted_idx.item()

# 3. 메인 실행 블록
if __name__ == '__main__':
    # --- 설정값 ---
    MODEL_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\model\binary_audio_cnn_model.pth' # 훈련 스크립트에서 저장한 모델 파일
    TEST_DATA_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv\validation_data'
    # val 폴더에 있는 정답 CSV 파일 경로
    TEST_CSV_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv\validation_data\REFERENCE2.csv' 
    # -----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
    except FileNotFoundError:
        print(f"오류: '{MODEL_PATH}' 파일을 찾을 수 없습니다.")
        exit()

    # 모델 파일에서 설정값 불러오기
    CLASS_NAMES = checkpoint['class_names']
    MAX_LEN = checkpoint['max_len']
    NUM_CLASSES = len(CLASS_NAMES)
    
    print(f"모델에서 불러온 설정: 클래스={CLASS_NAMES}, 최대 길이={MAX_LEN}")

    # 모델 초기화 및 가중치 로드
    model = AudioCNN(NUM_CLASSES).to(device)
    
    INPUT_FEATURE_HEIGHT = 53 
    dummy_input = torch.randn(1, 1, INPUT_FEATURE_HEIGHT, MAX_LEN).to(device)
    model(dummy_input)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"'{MODEL_PATH}'에서 모델 로딩 완료.")

    # --- [추가된 부분] 정답 라벨 로드 ---
    try:
        # CSV 파일에 헤더가 없다고 가정하고 읽음
        df_labels = pd.read_csv(TEST_CSV_PATH, header=None)
        # 빠른 조회를 위해 딕셔너리로 변환 (key: 파일명, value: 라벨)
        true_labels = {os.path.splitext(str(row[0]))[0]: row[1] for index, row in df_labels.iterrows()}
        print(f"'{TEST_CSV_PATH}'에서 정답 라벨 로딩 완료.")
    except FileNotFoundError:
        print(f"오류: 정답 CSV 파일 '{TEST_CSV_PATH}'를 찾을 수 없습니다.")
        true_labels = None
    # ------------------------------------

    print(f"\n--- '{TEST_DATA_PATH}' 폴더에 대한 추론 시작 ---")
    
    correct_predictions = 0
    total_files = 0

    if not os.path.exists(TEST_DATA_PATH) or not os.listdir(TEST_DATA_PATH):
         print(f"경고: '{TEST_DATA_PATH}' 폴더가 비어있거나 존재하지 않습니다.")
    else:
        for filename in os.listdir(TEST_DATA_PATH):
            if filename.endswith('.npy'):
                total_files += 1
                file_path = os.path.join(TEST_DATA_PATH, filename)
                
                predicted_index = predict_single_file(model, file_path, MAX_LEN, device)
                predicted_class_label = CLASS_NAMES[predicted_index]
                
                # 정답률 계산
                if true_labels:
                    base_filename = os.path.splitext(filename)[0]
                    if base_filename in true_labels:
                        true_label = true_labels[base_filename]
                        print(f"파일: {filename}, 예측: {predicted_class_label}, 정답: {true_label}", end="")
                        if int(predicted_class_label) == int(true_label):
                            correct_predictions += 1
                            print(" -> O")
                        else:
                            print(" -> X")
                    else:
                        print(f"파일: {filename}, 예측: {predicted_class_label} (정답 라벨 없음)")
                else:
                    print(f"파일: {filename}, 예측 라벨: {predicted_class_label}")

    # --- [추가된 부분] 최종 정확도 출력 ---
    if true_labels and total_files > 0:
        accuracy = (correct_predictions / total_files) * 100
        print("\n--- 추론 완료 ---")
        print(f"총 {total_files}개 파일 중 {correct_predictions}개 정답")
        print(f"정확도: {accuracy:.2f}%")
    else:
        print("\n--- 추론 완료 ---")
        if not true_labels:
            print("정답 파일이 없어 정확도를 계산할 수 없습니다.")

