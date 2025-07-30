import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# import librosa # WAV 파일 로드에는 사용되지 않지만, 더미 WAV 생성 시 필요할 수 있음
# import pywt # 웨이블릿 변환에는 사용되지 않지만, 더미 WAV 생성 시 사용될 수 있음
import soundfile as sf  # 더미 파일 생성을 위함

# GPU 사용 가능 여부 확인
# CUDA (NVIDIA GPU) 사용 가능하면 'cuda', 아니면 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 수정된 load_labels_and_data_paths 함수 ---
def load_labels_and_data_paths(wavelet_data_path, label_file_path):
    """
    라벨 파일과 웨이블릿 변환된 TXT 데이터 경로를 로드하고 매칭합니다.

    Args:
        wavelet_data_path (str): 웨이블릿 변환된 TXT 파일들이 있는 디렉토리 경로 (예: 'Wavelet_Transformed_Data').
        label_file_path (str): 라벨 정보가 담긴 CSV 또는 TXT 파일 경로 (예: 'REFERENCE.csv').

    Returns:
        tuple: (list of wavelet TXT file paths, list of corresponding labels)
               또는 (None, None) 오류 발생 시.
    """
    data_file_paths = []
    labels = []
    skipped_files_count = 0

    if not os.path.exists(label_file_path):
        print(
            f"오류: 라벨 파일 '{label_file_path}'을 찾을 수 없습니다. 경로를 확인해주세요."
        )
        return None, None

    try:
        df_labels = pd.read_csv(
            label_file_path, header=None, names=["filename_prefix", "label"]
        )
        print(f"'{label_file_path}'에서 {len(df_labels)}개의 라벨 로드 완료.")

        for index, row in df_labels.iterrows():
            # 파일명 접두사를 사용하여 웨이블릿 변환된 TXT 파일명 구성
            # 예: 'a0001' -> 'a0001_wavelet.txt'
            txt_filename = f"{row['filename_prefix']}_wavelet.txt"
            txt_file_path = os.path.join(wavelet_data_path, txt_filename)

            if os.path.exists(txt_file_path):
                data_file_paths.append(txt_file_path)
                labels.append(row["label"])
            else:
                # print(f"경고: 웨이블릿 TXT 파일 '{txt_file_path}'을 찾을 수 없습니다. 건너뜜.")
                skipped_files_count += 1

    except pd.errors.EmptyDataError:
        print(f"오류: 라벨 파일 '{label_file_path}'이 비어 있습니다.")
        return None, None
    except Exception as e:
        print(f"라벨 파일 로드 또는 처리 중 오류 발생: {e}")
        return None, None

    if skipped_files_count > 0:
        print(
            f"총 {skipped_files_count}개의 웨이블릿 TXT 파일을 찾을 수 없어 건너뛰었습니다. 경로를 다시 확인해주세요."
        )
    return data_file_paths, labels


# --- 이전 코드와 동일한 preprocess_labels 함수 ---
def preprocess_labels(labels):
    """
    라벨 (문자열 또는 숫자)을 수치형 (인코딩) 및 원-핫 인코딩 형식으로 변환합니다.
    "1이 정상, 0이 비정상" 규칙을 따릅니다.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    mapped_class_names = [
        f"Class {cls_val} ({'비정상' if cls_val == 0 else '정상'})"
        for cls_val in label_encoder.classes_
    ]

    print(f"원본 라벨 종류: {np.unique(labels)}")
    print(f"내부 인코딩된 라벨 종류: {np.unique(encoded_labels)}")
    print(f"매핑된 클래스 이름 (0:비정상, 1:정상): {mapped_class_names}")

    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_labels = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1))
    print(f"원-핫 인코딩된 라벨 형태: {onehot_labels.shape}")

    return encoded_labels, onehot_labels, label_encoder, mapped_class_names


# --- PyTorch Custom Dataset 정의 (웨이블릿 데이터용) ---
class WaveletTextDataset(Dataset):
    def __init__(self, data_file_paths, labels, max_sequence_length):
        self.data_file_paths = data_file_paths
        # 라벨은 CrossEntropyLoss를 위해 long 타입으로 저장
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_sequence_length = max_sequence_length
        self.cached_data = {}  # 데이터를 미리 로드하여 캐싱

        print(f"데이터셋 초기화 중. 총 {len(data_file_paths)}개 파일 처리 예상...")
        for i, path in enumerate(self.data_file_paths):
            try:
                y_data = np.loadtxt(path, dtype=np.float32)

                if len(y_data) < self.max_sequence_length:
                    pad_width = self.max_sequence_length - len(y_data)
                    y_padded = np.pad(
                        y_data, (0, pad_width), "constant", constant_values=0
                    )
                else:
                    y_padded = y_data[: self.max_sequence_length]

                # PyTorch Conv1D는 (batch_size, channels, sequence_length) 형태를 기대
                # 여기서는 channels=1 이므로 (1, sequence_length) 형태로 저장
                self.cached_data[i] = torch.tensor(
                    y_padded.reshape(1, -1), dtype=torch.float32
                )

            except Exception as e:
                print(
                    f"  데이터셋 로드 오류: '{os.path.basename(path)}' - {e}. 해당 인덱스 건너뜜."
                )
                self.cached_data[i] = None  # 오류 발생 파일은 None으로 표시

        # 오류 없이 로드된 데이터만 사용하도록 필터링
        self.valid_indices = [
            i for i, data in self.cached_data.items() if data is not None
        ]
        print(f"데이터셋 로드 완료. 유효한 파일 수: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        data = self.cached_data[original_idx]
        label = self.labels[original_idx]
        return data, label


# --- PyTorch 1D CNN 모델 정의 ---
class OneDCNN(nn.Module):
    def __init__(self, num_classes, input_length):  # input_length를 파라미터로 받음
        super(OneDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv1d는 (in_channels, out_channels, kernel_size)
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        # Flattening 전 feature map 크기 계산
        self._calculate_flatten_size(input_length)

        self.fc_layers = nn.Sequential(
            nn.Linear(self._flatten_size, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes),
            # CrossEntropyLoss는 내부적으로 Softmax를 포함하므로 최종 Softmax는 필요 없음
        )

    def _calculate_flatten_size(self, input_length):
        # 더미 텐서를 만들어서 Conv 층을 통과시켜 출력 크기를 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)  # (batch, channels, length)
            dummy_output = self.conv_layers(dummy_input)
            self._flatten_size = (
                dummy_output.numel() // dummy_output.shape[0]
            )  # 배치 차원 제외

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# --- 사용 예시 (메인 코드) ---
if __name__ == "__main__":
    # 🚨🚨🚨 실제 데이터셋 경로와 REFERENCE.csv 경로로 변경해야 합니다 🚨🚨🚨
    # ==========================================================경로 지정==========================================================
    wavelet_transformed_data_folder = (
        r"D:\MDO\heartbeat\Wavelet/Wavelet_Transformed_Data"
    )
    label_file_path = r"D:\MDO\heartbeat\Wavelet/Wavelet_Transformed_Data/REFERENCE.csv"

    # ❗❗❗ 테스트를 위한 더미 파일 및 폴더 생성 (실제 사용 시에는 이 부분 삭제) ❗❗❗
    if not os.path.exists(wavelet_transformed_data_folder) or not os.path.exists(
        label_file_path
    ):
        print("\n--- 테스트를 위해 더미 웨이블릿 변환 데이터 폴더 및 파일 생성 ---")
        os.makedirs(wavelet_transformed_data_folder, exist_ok=True)
        print(f"더미 폴더 '{wavelet_transformed_data_folder}' 생성 완료.")

        dummy_labels_data = [
            ("a0001", 1),  # 정상
            ("a0002", 0),  # 비정상
            ("b0001", 1),  # 정상
            ("c0001", 1),  # 정상
            ("c0002", 0),  # 비정상
            ("a0003", 1),
            ("b0002", 0),
            ("c0003", 1),
            ("a0004", 0),
            ("b0003", 1),
            ("d0001", 1),
            ("d0002", 0),
            ("e0001", 1),
            ("f0001", 0),  # 추가 더미 데이터
        ]
        dummy_df = pd.DataFrame(dummy_labels_data, columns=["filename_prefix", "label"])
        dummy_df.to_csv(label_file_path, index=False, header=False)
        print(f"더미 '{label_file_path}' 생성 완료 (라벨: 0=비정상, 1=정상).")

        # 더미 웨이블릿 변환 TXT 파일 생성
        # 실제 웨이블릿 변환을 거친 데이터는 길이가 길 수 있으므로, 적절한 더미 길이 설정
        for prefix, _ in dummy_labels_data:
            dummy_txt_path = os.path.join(
                wavelet_transformed_data_folder, f"{prefix}_wavelet.txt"
            )
            # 더미 데이터 길이를 무작위로 하여 패딩/자르기 테스트 (실제 웨이블릿 데이터는 다를 수 있음)
            dummy_data_length = np.random.randint(
                1000, 5000
            )  # 예를 들어, 1000~5000개의 계수
            dummy_data = np.random.uniform(-1, 1, dummy_data_length).astype(np.float32)
            np.savetxt(dummy_txt_path, dummy_data, fmt="%f", delimiter="\n")
        print("--- 더미 웨이블릿 TXT 파일 생성 완료. 이제 실행할 수 있습니다. ---")
    # ❗❗❗ 더미 파일 생성 끝 ❗❗❗

    # 1. 라벨 및 웨이블릿 변환된 데이터 경로 로드
    data_file_paths, raw_labels = load_labels_and_data_paths(
        wavelet_transformed_data_folder, label_file_path
    )

    if data_file_paths and raw_labels:
        print("\n--- 라벨 전처리 ---")
        # encoded_labels (정수형)와 onehot_labels (원-핫) 모두 얻습니다.
        encoded_labels, onehot_labels, label_encoder_obj, class_names_mapping = (
            preprocess_labels(raw_labels)
        )

        print(f"\n로드된 유효 웨이블릿 TXT 파일 수: {len(data_file_paths)}")
        print(f"로드된 유효 라벨 수: {len(onehot_labels)}")

        # 2. 훈련 세트와 검증 세트로 분리
        # PyTorch Dataset에 전달하기 위해 정수형 라벨(encoded_labels)을 사용합니다.
        X_train_paths, X_val_paths, y_train_encoded, y_val_encoded = train_test_split(
            data_file_paths,
            encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels,
        )

        print(f"\n훈련 세트 크기 (경로): {len(X_train_paths)}")
        print(f"검증 세트 크기 (경로): {len(X_val_paths)}")

        # 3. PyTorch Dataset 및 DataLoader 준비
        # 🚨🚨🚨 MAX_SEQUENCE_LENGTH_WAVELET은 실제 웨이블릿 변환 데이터의 길이를 기반으로 설정해야 합니다.
        # 모든 웨이블릿 계수의 길이를 분석하여 적절한 값을 설정하세요.
        # 예를 들어, 웨이블릿 변환된 데이터의 평균 길이 또는 최대 길이를 고려합니다.
        MAX_SEQUENCE_LENGTH_WAVELET = (
            2000  # 예시: 웨이블릿 계수가 이 정도 길이를 가질 수 있다고 가정
        )

        print(f"\n--- 훈련 데이터셋 생성 (웨이블릿 데이터 사용) ---")
        # WaveletTextDataset에 정수형 라벨(y_train_encoded)을 전달
        train_dataset = WaveletTextDataset(
            X_train_paths, y_train_encoded, MAX_SEQUENCE_LENGTH_WAVELET
        )
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=0
        )  # num_workers는 환경에 따라 조정 (Windows에서는 0 권장)
        print(f"훈련 데이터셋 크기: {len(train_dataset)}")

        print(f"\n--- 검증 데이터셋 생성 (웨이블릿 데이터 사용) ---")
        # WaveletTextDataset에 정수형 라벨(y_val_encoded)을 전달
        val_dataset = WaveletTextDataset(
            X_val_paths, y_val_encoded, MAX_SEQUENCE_LENGTH_WAVELET
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        print(f"검증 데이터셋 크기: {len(val_dataset)}")

        # 4. PyTorch 1D CNN 모델 구축 및 학습
        num_classes = onehot_labels.shape[
            1
        ]  # num_classes는 원-핫 인코딩된 라벨의 컬럼 수
        # 모델 생성 시 MAX_SEQUENCE_LENGTH_WAVELET을 전달하여 Flatten 크기 계산에 사용
        model = OneDCNN(
            num_classes=num_classes, input_length=MAX_SEQUENCE_LENGTH_WAVELET
        ).to(device)

        # 옵티마이저와 손실 함수 정의
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # CrossEntropyLoss는 타겟이 클래스 인덱스 (정수, Long 타입)여야 하며, 내부적으로 Softmax 포함
        criterion = nn.CrossEntropyLoss()

        print("\n--- PyTorch 1D CNN 모델 학습 시작 (웨이블릿 변환 데이터) ---")
        num_epochs = 10  # 에포크 수 조정

        for epoch in range(num_epochs):
            model.train()  # 훈련 모드
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # 기울기 초기화
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # outputs는 float, labels는 long
                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(
                    outputs.data, 1
                )  # 예측은 outputs에서 가장 높은 확률의 인덱스
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = 100 * correct_train / total_train

            # 검증 단계
            model.eval()  # 평가 모드
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():  # 기울기 계산 비활성화
                for inputs_val, labels_val in val_loader:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(
                        device
                    )
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)

                    val_loss += loss_val.item() * inputs_val.size(0)
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()

            val_epoch_loss = val_loss / len(val_dataset)
            val_epoch_acc = 100 * correct_val / total_val

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%"
            )

        print("PyTorch 1D CNN 모델 학습 완료 (웨이블릿 변환 데이터).")

        # --- 모델 저장 ---
        # ==========================================================경로 지정==========================================================
        model_save_dir = "./Wavelet/model"  # 저장 경로 수정
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path_wavelet = os.path.join(model_save_dir, "wavelet_cnn_model.pth")

        # 모델의 상태 사전(state_dict) 저장
        torch.save(model.state_dict(), model_save_path_wavelet)
        print(
            f"\n모델이 다음 경로에 성공적으로 저장되었습니다: '{os.path.abspath(model_save_path_wavelet)}'"
        )

        # --- 저장된 모델을 다시 로드하는 예시 (확인용) ---
        # loaded_model = OneDCNN(num_classes=num_classes, input_length=MAX_SEQUENCE_LENGTH_WAVELET).to(device)
        # loaded_model.load_state_dict(torch.load(model_save_path_wavelet))
        # loaded_model.eval() # 평가 모드로 전환 (Dropout 등이 비활성화됨)
        # print(f"\n모델이 '{model_save_path_wavelet}'에서 성공적으로 로드되었습니다.")
        # # 간단한 테스트 (예: 한 배치 가져와서 추론)
        # # with torch.no_grad():
        # #     sample_input, sample_label = next(iter(val_loader))
        # #     sample_input = sample_input.to(device)
        # #     output_test = loaded_model(sample_input)
        # #     print(f"로드된 모델 출력 형태: {output_test.shape}")

    else:
        print(
            "라벨 또는 데이터 경로 로드에 실패했습니다. 다음 단계로 진행할 수 없습니다."
        )
