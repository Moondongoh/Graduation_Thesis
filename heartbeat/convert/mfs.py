# ============================================================
# 1. Mel-Spectrogram 이미지 변환 파트
# - librosa를 이용해 .wav 파일을 Mel-Spectrogram으로 변환
# - 변환된 스펙트로그램을 matplotlib으로 이미지(.png)로 저장
# - 결과 이미지 저장 위치: ./mels_images/validation/
# ============================================================

# import os
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# def create_melspectrogram(wav_path, save_path):
#     y, sr = librosa.load(wav_path, sr=None)
#     mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     mels_db = librosa.power_to_db(mels, ref=np.max)

#     plt.figure(figsize=(2.24, 2.24))
#     librosa.display.specshow(mels_db, sr=sr, x_axis=None, y_axis=None)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # 폴더 설정
# wav_dir = "./classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/validation"
# save_dir = "./mels_images/validation"
# os.makedirs(save_dir, exist_ok=True)

# # 변환 실행
# for wav_file in tqdm(os.listdir(wav_dir)):
#     if wav_file.endswith(".wav"):
#         wav_path = os.path.join(wav_dir, wav_file)
#         save_path = os.path.join(save_dir, wav_file.replace(".wav", ".png"))
#         try:
#             create_melspectrogram(wav_path, save_path)
#         except Exception as e:
#             print(f"{wav_file} 변환 실패: {e}")

# ============================================================
# 2. REFERENCE.csv 라벨 수정 파트
# - -1을 0으로 변경하고 새로운 파일로 저장
# - 결과 파일: REFERENCE_binary.csv
# ============================================================

import pandas as pd

# 파일 경로
csv_path = "./classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/validation/REFERENCE.csv"

# CSV 불러오기
df = pd.read_csv(csv_path, header=None)
df.columns = ["file_name", "label"]

# -1을 0으로 변경
df["label"] = df["label"].replace(-1, 0)

df.to_csv("REFERENCE_binary.csv", index=False, header=False)
