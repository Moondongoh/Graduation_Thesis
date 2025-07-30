import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
)  # 라벨 전처리를 위함 (평가 시 필요)


# --- load_labels_and_data_paths 함수 (이전과 동일) ---
def load_labels_and_data_paths(data_base_path, label_file_path):
    """
    라벨 파일과 1D TXT 데이터 경로를 로드하고 매칭합니다.
    Args:
        data_base_path (str): 1D TXT 파일들이 있는 디렉토리 경로.
        label_file_path (str): 라벨 정보가 담긴 CSV 또는 TXT 파일 경로.
    Returns:
        tuple: (list of TXT file paths, list of corresponding labels)
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
        print(f"'{label_file_path}'에서 {len(df_labels)}개의 라벨 정보를 로드했습니다.")

        for index, row in df_labels.iterrows():
            # 원본 1D 데이터를 사용하므로 '_data.txt' 확장자를 사용합니다.
            txt_filename = f"{row['filename_prefix']}_data.txt"
            txt_file_path = os.path.join(data_base_path, txt_filename)

            if os.path.exists(txt_file_path):
                data_file_paths.append(txt_file_path)
                labels.append(row["label"])
            else:
                print(
                    f"경고: 매칭되는 TXT 파일 '{txt_file_path}'을 찾을 수 없습니다. 건너뜁니다."
                )
                skipped_files_count += 1

    except pd.errors.EmptyDataError:
        print(f"오류: 라벨 파일 '{label_file_path}'이 비어 있습니다.")
        return None, None
    except Exception as e:
        print(f"라벨 파일 로드 또는 처리 중 오류 발생: {e}")
        return None, None

    if skipped_files_count > 0:
        print(
            f"총 {skipped_files_count}개의 TXT 파일을 찾을 수 없어 건너뛰었습니다. 경로를 다시 확인해주세요."
        )
    return data_file_paths, labels


# --- preprocess_labels 함수 (이전과 동일) ---
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


# --- load_data_for_cnn 함수 (이전과 동일) ---
def load_data_for_cnn(data_file_paths, max_sequence_length):
    """
    주어진 TXT 파일 경로 리스트에서 숫자 데이터를 로드하고 CNN 입력을 위해 전처리합니다.
    모든 데이터 시퀀스의 길이를 max_sequence_length로 통일합니다 (패딩 또는 자르기).
    """
    processed_data = []

    for i, path in enumerate(data_file_paths):
        try:
            y_data = np.loadtxt(path)

            if len(y_data) < max_sequence_length:
                pad_width = max_sequence_length - len(y_data)
                y_padded = np.pad(y_data, (0, pad_width), "constant")
            else:
                y_padded = y_data[:max_sequence_length]

            processed_data.append(y_padded.reshape(-1, 1))

            if i % 100 == 0:
                print(
                    f"  {i+1}/{len(data_file_paths)} 파일 처리 완료: {os.path.basename(path)}"
                )

        except Exception as e:
            print(
                f"  오류: '{os.path.basename(path)}' 처리 중 문제 발생 - {e}. 해당 파일 건너뜀."
            )

    return np.array(processed_data)


if __name__ == "__main__":
    # 🚨🚨🚨 실제 모델 경로 및 검증 데이터 경로를 환경에 맞게 설정해주세요 🚨🚨🚨

    # 1. 학습된 모델 파일 경로
    # 모델 학습 시 저장된 정확한 .keras 파일 경로를 지정해야 합니다.
    # 예: 'D:/MDO/heartbeat/Wavelet/saved_models/wavelet_cnn_model.keras'
    # ==========================================================경로 지정==========================================================
    model_path = r".\Wavelet\model\wavelet_cnn_model.keras"

    # 2. 추론할 원본 1D 데이터 폴더 경로
    # './Wavelet/validation/Validation_Tansformed_Data' 에 원본 1D TXT 파일들이 있다고 가정합니다.
    # ==========================================================경로 지정==========================================================
    validation_data_folder = r"D:\MDO\heartbeat\Wavelet\Validation_Transformed_Data"

    # 3. 검증 데이터에 대한 라벨 파일 경로 (평가 시 필요)
    # 일반적으로 REFERENCE.csv는 데이터셋의 최상위 폴더에 있습니다.
    # 만약 검증 데이터셋만을 위한 별도의 REFERENCE.csv가 있다면 해당 경로를 지정하세요.
    # ==========================================================경로 지정==========================================================
    validation_label_file_path = (
        r"D:\MDO\heartbeat\Wavelet/Validation_Transformed_Data\REFERENCE.csv"
    )

    # 4. 학습 시 사용했던 MAX_SEQUENCE_LENGTH 값 (가장 중요!)
    # 🚨🚨🚨 이 값을 모델 학습 시 사용한 정확한 값으로 다시 변경합니다! 🚨🚨🚨
    # 현재 로드된 모델은 2000 길이의 시퀀스로 학습되었습니다.
    MAX_SEQUENCE_LENGTH_WAVELET = 2000  # 🚨🚨🚨 다시 2000으로 수정됨! 🚨🚨🚨

    print(f"\n--- 모델 로드 시작: {model_path} ---")
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        print(f"모델이 '{model_path}'에서 성공적으로 로드되었습니다.")
        loaded_model.summary()  # 모델 로드 후 summary를 다시 출력하여 input shape 확인
    except Exception as e:
        print(f"오류: 모델 로드 실패 - {e}")
        print("모델 경로를 다시 확인하거나, 모델이 손상되지 않았는지 확인해주세요.")
        exit()  # 모델 로드 실패 시 스크립트 종료

    print(f"\n--- 검증 데이터 로드 시작: {validation_data_folder} ---")
    # load_labels_and_data_paths 함수는 이제 _data.txt 파일을 찾습니다.
    val_data_file_paths, val_raw_labels = load_labels_and_data_paths(
        validation_data_folder, validation_label_file_path
    )

    if val_data_file_paths and val_raw_labels:
        print("\n--- 검증 데이터 라벨 전처리 ---")
        (
            val_encoded_labels,
            val_onehot_labels,
            val_label_encoder_obj,
            val_class_names_mapping,
        ) = preprocess_labels(val_raw_labels)

        print(f"\n로드된 유효 검증 원본 1D TXT 파일 수: {len(val_data_file_paths)}")
        print(f"로드된 유효 검증 라벨 수: {len(val_onehot_labels)}")

        print(f"\n--- 검증 데이터 전처리 (CNN 입력 형태) ---")
        X_val_processed = load_data_for_cnn(
            val_data_file_paths, MAX_SEQUENCE_LENGTH_WAVELET
        )
        print(f"전처리된 검증 데이터 형태: {X_val_processed.shape}")

        if X_val_processed.shape[0] == 0:
            print(
                "처리된 검증 데이터가 없습니다. 파일 경로 및 이름 패턴을 다시 확인해주세요."
            )
        else:
            print("\n--- 추론 수행 ---")
            predictions = loaded_model.predict(X_val_processed)
            predicted_classes = np.argmax(
                predictions, axis=1
            )  # 가장 높은 확률을 가진 클래스 인덱스

            # 예측 결과 및 실제 라벨 매핑 (0:비정상, 1:정상)
            print("\n--- 추론 결과 ---")
            for i, pred_class_idx in enumerate(predicted_classes):
                true_label = val_raw_labels[i]
                predicted_label = val_label_encoder_obj.inverse_transform(
                    [pred_class_idx]
                )[0]

                # 라벨이 '0'이면 '비정상', '1'이면 '정상'으로 매핑하여 출력
                true_label_str = "정상" if true_label == 1 else "비정상"
                predicted_label_str = "정상" if predicted_label == 1 else "비정상"

                print(
                    f"파일: {os.path.basename(val_data_file_paths[i])} | 실제 라벨: {true_label_str} | 예측 라벨: {predicted_label_str} (확률: {predictions[i]})"
                )

            # 모델 평가 (선택 사항)
            print("\n--- 모델 성능 평가 ---")
            loss, accuracy = loaded_model.evaluate(
                X_val_processed, val_onehot_labels, verbose=0
            )
            print(f"검증 데이터 손실: {loss:.4f}")
            print(f"검증 데이터 정확도: {accuracy:.4f}")

    else:
        print("검증 데이터 로드에 실패했습니다. 추론을 진행할 수 없습니다.")
