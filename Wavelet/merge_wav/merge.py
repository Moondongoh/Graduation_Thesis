import os
import shutil
import pandas as pd

# 1. 절대 경로 설정
base_dir = r"D:\MDO\heartbeat\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
train_dirs = [f"training-{c}" for c in "abcdef"]
val_dir = os.path.join(base_dir, "validation")

# 2. 결과 저장 경로
output_wav_dir = "Dataset/wav_filtered"
output_csv_path = "Dataset/REFERENCE_filtered.csv"
os.makedirs(output_wav_dir, exist_ok=True)

# 3. validation 폴더에 있는 파일 목록 추출 (a0001.wav → a0001)
val_files = set(f.replace(".wav", "") for f in os.listdir(val_dir) if f.endswith(".wav"))

# 4. 전체 레이블 저장용 리스트
filtered_rows = []

for train_folder in train_dirs:
    folder_path = os.path.join(base_dir, train_folder)
    reference_file = os.path.join(folder_path, "REFERENCE.csv")

    # 해당 폴더의 레이블 csv 불러오기
    if not os.path.exists(reference_file):
        print(f"[!] REFERENCE.csv not found in {train_folder}")
        continue

    df = pd.read_csv(reference_file, header=None, names=["file_name", "label"])

    for fname in os.listdir(folder_path):
        if fname.endswith(".wav"):
            file_id = fname.replace(".wav", "")
            if file_id in val_files:
                continue  # validation과 겹치면 제외

            # wav 복사
            src_path = os.path.join(folder_path, fname)
            dst_path = os.path.join(output_wav_dir, fname)
            shutil.copyfile(src_path, dst_path)

            # 레이블 찾기
            label_row = df[df["file_name"] == file_id]
            if not label_row.empty:
                filtered_rows.append(label_row.iloc[0])
            else:
                print(f"[!] 레이블 누락: {file_id} (in {train_folder})")

# 5. 통합된 레이블 저장
filtered_df = pd.DataFrame(filtered_rows)
filtered_df.to_csv(output_csv_path, index=False, header=False)

print(f"\n✅ 총 {len(filtered_df)}개의 wav와 레이블을 저장했습니다.")
