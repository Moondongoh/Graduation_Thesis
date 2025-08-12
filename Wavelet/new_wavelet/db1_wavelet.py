import os
import numpy as np
import pywt
import soundfile as sf
import pandas as pd
from tqdm import tqdm


def extract_wavelet_features(signal, wavelet="db1", level=5, segment_length=512):
    """
    주어진 1D 신호를 512 단위로 분할하고, 각 분할에 대해 5레벨 웨이블릿 변환을 수행하여
    각 계수에 대한 통계적 특징(평균, 표준편차, 최대값, 최소값, 중앙값)을 추출한다.

    파라미터:
    - wavelet: 'db1' (Haar 웨이블릿)
    - level: 5
    - segment_length: 512

    반환값: (세그먼트 수, 30) 크기의 2D NumPy 배열
    """
    num_segments = len(signal) // segment_length
    features = []

    for i in range(num_segments):
        segment = signal[i * segment_length : (i + 1) * segment_length]
        coeffs = pywt.wavedec(segment, wavelet, level=level)

        segment_features = []
        for c in coeffs:
            segment_features.extend(
                [np.mean(c), np.std(c), np.max(c), np.min(c), np.median(c)]
            )
        features.append(segment_features)

    return np.array(features)


def get_feature_column_names():
    """
    컬럼 이름 생성: cA5~cD1 각각에 대해 mean, std, max, min, median
    """
    levels = ["cA5", "cD5", "cD4", "cD3", "cD2", "cD1"]
    stats = ["mean", "std", "max", "min", "median"]
    return [f"{level}_{stat}" for level in levels for stat in stats]


# ===== 경로 설정 =====
# 현재 데이터셋이 위치한 경로를 지정합니다.
# input_dir = r"D:\MDO\heartbeat\Dataset"
# # output_dir을 'Features_db1'과 같이 변경하여 db1 결과와 db4 결과를 구분할 수 있습니다.
# output_dir = r"D:\MDO\heartbeat\Features_db1"
# label_file_path = os.path.join(input_dir, "REFERENCE_filtered.csv")

# 현재 데이터셋이 위치한 경로를 지정합니다.
base_dir = r"D:\MDO\heartbeat\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
input_dir = os.path.join(base_dir, "validation")
output_dir = r"D:\MDO\heartbeat\validation_Features_db1"
label_file_path = os.path.join(input_dir, "REFERENCE.csv")


# 출력 디렉토리가 없으면 생성합니다.
os.makedirs(output_dir, exist_ok=True)

# 라벨 파일 로드
try:
    labels_df = pd.read_csv(label_file_path, header=None, names=["filename", "label"])
    labels_df["filename"] = labels_df["filename"].apply(lambda x: x.split(".")[0])
    labels_dict = labels_df.set_index("filename")["label"].to_dict()
    print("✅ 라벨 파일이 성공적으로 로드되었습니다.")
except FileNotFoundError:
    print(f"❌ 라벨 파일이 존재하지 않습니다: {label_file_path}")
    labels_dict = {}
except Exception as e:
    print(f"❌ 라벨 파일 로드 중 오류 발생: {e}")
    labels_dict = {}

# ===== .wav 파일 처리 및 저장 =====
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        try:
            signal, sr = sf.read(filepath)
            # wavelet 인자를 'db1'로 지정합니다.
            features = extract_wavelet_features(signal, wavelet="db1")

            file_key = os.path.splitext(filename)[0]
            label_value = labels_dict.get(file_key)

            if label_value is None:
                print(f"⚠️ {filename}에 대한 라벨을 찾을 수 없습니다. 건너뜁니다.")
                continue

            df = pd.DataFrame(features, columns=get_feature_column_names())
            df["label"] = label_value

            out_name = os.path.splitext(filename)[0] + ".csv"
            out_path = os.path.join(output_dir, out_name)
            df.to_csv(out_path, index=False)
            print(f"✅ {out_name} 저장 완료. 라벨: {label_value}")

        except Exception as e:
            print(f"❌ {filename} 처리 중 오류 발생: {e}")
