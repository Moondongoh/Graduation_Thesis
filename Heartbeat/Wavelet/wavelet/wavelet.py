import os
import numpy as np
import pandas as pd
import pywt
import librosa
from tqdm import tqdm

def extract_wavelet_statistical_features(signal, wavelet='db4', level=5, segment_length=512):
    """
    1D 신호를 세그먼트로 분할하고, 각 세그먼트의 웨이블릿 계수로부터
    통계적 특징을 추출합니다.
    """
    num_segments = len(signal) // segment_length
    features = []

    # 웨이블릿 계수 이름 만들기
    coeff_names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    feature_columns = []
    for coeff_name in coeff_names:
        for stat in ['mean', 'std', 'max', 'min', 'median', 'energy']:
            feature_columns.append(f"{coeff_name}_{stat}")

    for i in range(num_segments):
        segment = signal[i * segment_length : (i + 1) * segment_length]
        coeffs = pywt.wavedec(segment, wavelet, level=level)

        segment_features = []
        for c in coeffs:
            stats = [
                np.mean(c),
                np.std(c),
                np.max(c),
                np.min(c),
                np.median(c),
                np.sum(c**2)   # energy 추가
            ]
            segment_features.extend(stats)
        features.append(segment_features)

    return pd.DataFrame(features, columns=feature_columns)


def process_folder_to_csv(folder_path, output_csv_path, wavelet='db4', level=5, segment_length=512):
    """
    폴더 안의 모든 WAV 파일을 불러와 웨이블릿 특징을 추출하고
    하나의 CSV로 저장합니다.
    """
    all_features = []

    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not wav_files:
        print(f"❌ '{folder_path}' 폴더에 wav 파일이 없습니다.")
        return

    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        wav_path = os.path.join(folder_path, wav_file)

        try:
            y, sr = librosa.load(wav_path, sr=None)
            df_features = extract_wavelet_statistical_features(y, wavelet, level, segment_length)
            df_features['file'] = wav_file  # 파일명 추가
            all_features.append(df_features)
        except Exception as e:
            print(f"⚠️ {wav_file} 처리 중 오류 발생: {e}")

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)
        print(f"✅ 모든 특징 추출 완료, CSV 저장: '{output_csv_path}'")
    else:
        print("❌ 추출된 특징이 없습니다.")

# INPUT_FOLDER = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset/train'
# OUTPUT_CSV = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset/wavelet_train'

INPUT_FOLDER = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset/validation'
OUTPUT_CSV = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset/wavelet_validation'

process_folder_to_csv(INPUT_FOLDER, OUTPUT_CSV)
