# import os
# import numpy as np
# import pywt
# import soundfile as sf
# from tqdm import tqdm

# def extract_wavelet_features(signal, wavelet='db4', level=5, segment_length=512):
#     num_segments = len(signal) // segment_length
#     features = []

#     for i in range(num_segments):
#         segment = signal[i * segment_length : (i + 1) * segment_length]
#         coeffs = pywt.wavedec(segment, wavelet, level=level)

#         segment_features = []
#         for c in coeffs:
#             segment_features.extend([
#                 np.mean(c), 
#                 np.std(c), 
#                 np.max(c), 
#                 np.min(c), 
#                 np.median(c)
#             ])
#         features.append(segment_features)

#     return np.array(features)

# # ===== 경로 설정 =====
# #input_dir = r"D:\MDO\heartbeat\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"   # .wav 파일 폴더
# input_dir = r"D:\MDO\heartbeat\Dataset"   # .wav 파일 폴더
# output_dir = r"D:\MDO\heartbeat\csv_Features" # .npy 저장 폴더

# os.makedirs(output_dir, exist_ok=True)

# # ===== .wav 파일 처리 =====
# for filename in tqdm(os.listdir(input_dir)):
#     if filename.endswith(".wav"):
#         filepath = os.path.join(input_dir, filename)
#         try:
#             signal, sr = sf.read(filepath)  # wav 로드
#             features = extract_wavelet_features(signal)
            
#             # 저장 경로
#             out_name = os.path.splitext(filename)[0] + ".csv"
#             out_path = os.path.join(output_dir, out_name)
#             np.savetxt(out_path, features, delimiter=",", fmt="%.6f")

#         except Exception as e:
#             print(f"❌ {filename} 처리 중 오류 발생: {e}")


import os
import numpy as np
import pywt
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def extract_wavelet_features(signal, wavelet='db4', level=5, segment_length=512):
    """
    주어진 1D 신호를 512 단위로 분할하고, 각 분할에 대해 5레벨 웨이블릿 변환을 수행하여 
    각 계수에 대한 통계적 특징(평균, 표준편차, 최대값, 최소값, 중앙값)을 추출한다.

    반환값: (세그먼트 수, 30) 크기의 2D NumPy 배열
    """
    num_segments = len(signal) // segment_length
    features = []

    for i in range(num_segments):
        segment = signal[i * segment_length : (i + 1) * segment_length]
        coeffs = pywt.wavedec(segment, wavelet, level=level)

        segment_features = []
        for c in coeffs:
            segment_features.extend([
                np.mean(c), 
                np.std(c), 
                np.max(c), 
                np.min(c), 
                np.median(c)
            ])
        features.append(segment_features)

    return np.array(features)

def get_feature_column_names():
    """
    컬럼 이름 생성: cA5~cD1 각각에 대해 mean, std, max, min, median
    """
    levels = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']
    stats = ['mean', 'std', 'max', 'min', 'median']
    return [f"{level}_{stat}" for level in levels for stat in stats]

# ===== 경로 설정 =====
#input_dir = r"D:\MDO\heartbeat\Dataset"   # .wav 파일 폴더
#output_dir = r"D:\MDO\heartbeat\csv_Features" # .npy 저장 폴더

input_dir = r"D:\MDO\heartbeat\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/validation"   # .wav 파일 폴더
output_dir = r"D:\MDO\heartbeat\validation_Features" # .npy 저장 폴더

os.makedirs(output_dir, exist_ok=True)

# ===== .wav 파일 처리 및 저장 =====
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        try:
            signal, sr = sf.read(filepath)  # wav 로드
            features = extract_wavelet_features(signal)  # (세그먼트 수, 30)
            
            # CSV 저장
            df = pd.DataFrame(features, columns=get_feature_column_names())
            out_name = os.path.splitext(filename)[0] + ".csv"
            out_path = os.path.join(output_dir, out_name)
            df.to_csv(out_path, index=False)

        except Exception as e:
            print(f"❌ {filename} 처리 중 오류 발생: {e}")
