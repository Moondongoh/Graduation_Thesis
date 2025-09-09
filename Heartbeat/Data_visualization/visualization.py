import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

audio_dir = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\validation"
output_dir = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\new_val_wav_visualizations"
os.makedirs(output_dir, exist_ok=True)

max_amplitude = 0
for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=None)
        max_amplitude = max(max_amplitude, np.max(np.abs(y)))

max_amplitude = 0
for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=None)
        max_amplitude = max(max_amplitude, np.max(np.abs(y)))

print(f"전체 wav 파일 기준 최대 진폭: {max_amplitude}")

for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=None)
        if len(y) % 2 != 0:
            y = y[:-1]
        x_amp = y[0::2]
        y_amp = y[1::2]

        plt.figure(figsize=(6, 6))
        plt.scatter(x_amp, y_amp, s=4, alpha=0.6)
        plt.xlim(-max_amplitude, max_amplitude)
        plt.ylim(-max_amplitude, max_amplitude)
        plt.title(file)
        plt.xlabel("Amplitude (pair X)")
        plt.ylabel("Amplitude (pair Y)")
        save_path = os.path.join(output_dir, file.replace(".wav", ".png"))
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

print("진폭만 이용한 이미지 변환 완료!")
