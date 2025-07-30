# ============================================================
# 1. 학습에 사용될 파일 통합
# - training-a, b, c, d, e, f 폴더의 .wav 파일을 모두 하나의 폴더로 통합
# - REFERENCE.csv 파일을 하나로 통합하고 -1을 0으로 변경
# ============================================================

import os
import shutil
from tqdm import tqdm

# 원본 training 폴더들 위치
base_dir = r"E:/heartbeat/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
training_folders = [
    "training-a",
    "training-b",
    "training-c",
    "training-d",
    "training-e",
    "training-f",
]

# 결과 저장 위치
output_dir = os.path.join(base_dir, "all_training_wav")
os.makedirs(output_dir, exist_ok=True)

for folder in training_folders:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"{folder_path} 폴더가 존재하지 않습니다. 건너뜀.")
        continue

    for file in tqdm(os.listdir(folder_path), desc=f"복사 중: {folder}"):
        if file.endswith(".wav"):
            src = os.path.join(folder_path, file)
            dst = os.path.join(output_dir, file)
            try:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                else:
                    print(f"이미 존재함: {file} → 건너뜀")
            except PermissionError:
                print(f"권한 오류 발생: {file} → 복사 실패")


import pandas as pd

# 각 training-* 폴더에 있는 REFERENCE.csv 파일 병합
label_list = []
for folder in training_folders:
    label_path = os.path.join(base_dir, folder, "REFERENCE.csv")
    if os.path.exists(label_path):
        df = pd.read_csv(label_path, header=None)
        label_list.append(df)

# 모두 합치기
merged_labels = pd.concat(label_list, ignore_index=True)
merged_labels.columns = ["file_name", "label"]

# -1을 0으로 바꾸고 저장
merged_labels["label"] = merged_labels["label"].replace(-1, 0)
merged_labels.to_csv(os.path.join(base_dir, "REFERENCE.csv"), index=False, header=False)
