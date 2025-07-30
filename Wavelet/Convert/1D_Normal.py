import librosa
import numpy as np
import os
import pandas as pd  # REFERENCE.csv 처리를 위해 추가
import soundfile as sf  # 더미 WAV 파일 생성을 위함


def process_and_save_audio_data(wav_file_path, output_txt_path):
    """
    1차원 WAV 오디오 파일을 로드하고, 숫자 데이터를 TXT 파일로 저장합니다.

    Args:
        wav_file_path (str): 로드할 WAV 파일의 전체 경로.
        output_txt_path (str): 숫자 데이터를 저장할 TXT 파일의 전체 경로.
    """
    try:
        y, sr = librosa.load(wav_file_path, sr=None)
        np.savetxt(output_txt_path, y, fmt="%f", delimiter="\n")

    except FileNotFoundError:
        print(f"오류: '{wav_file_path}' 파일을 찾을 수 없습니다. 건너뜜.")
    except Exception as e:
        print(
            f"오류: '{os.path.basename(wav_file_path)}' 처리 중 문제 발생 - {e}. 건너뜜."
        )


def convert_all_wavs_to_1d_txt(input_wav_folder, output_txt_dir, label_file_path):
    """
    지정된 단일 입력 폴더 내의 모든 WAV 파일을 찾아 1D TXT 파일로 변환하여
    지정된 출력 디렉토리에 저장합니다.

    Args:
        input_wav_folder (str): 모든 WAV 파일이 직접 들어있는 단일 디렉토리 경로.
        output_txt_dir (str): 변환된 TXT 파일을 저장할 디렉토리 (예: 'Read_1D_Data/result').
        label_file_path (str): 모든 파일명 접두사를 알기 위한 REFERENCE.csv 파일 경로.
    """
    print(f"\n--- WAV 파일을 1D TXT로 변환 시작 ---")
    print(f"입력 WAV 폴더: {input_wav_folder}")
    print(f"출력 TXT 디렉토리: {output_txt_dir}")

    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_txt_dir, exist_ok=True)

    # REFERENCE.csv를 로드하여 모든 파일의 접두사를 얻습니다.
    # 이를 통해 어떤 파일들을 찾고 변환해야 하는지 명확히 하고, 출력 파일명을 결정합니다.
    if not os.path.exists(label_file_path):
        print(
            f"오류: 라벨 파일 '{label_file_path}'을 찾을 수 없습니다. 변환을 시작할 수 없습니다."
        )
        return

    try:
        df_labels = pd.read_csv(
            label_file_path, header=None, names=["filename_prefix", "label"]
        )
        # REFERENCE.csv에 있는 파일 접두사를 집합으로 만들어 빠른 조회를 가능하게 합니다.
        expected_prefixes = set(df_labels["filename_prefix"].tolist())
        total_expected_files = len(expected_prefixes)
        print(f"'{label_file_path}'에서 {total_expected_files}개의 파일 접두사 로드.")
    except Exception as e:
        print(
            f"오류: 라벨 파일 '{label_file_path}' 로드 중 문제 발생 - {e}. 변환을 시작할 수 없습니다."
        )
        return

    processed_count = 0
    skipped_count = 0

    # 입력 WAV 폴더 내의 모든 파일을 순회
    for filename in os.listdir(input_wav_folder):
        if filename.endswith(".wav"):
            # WAV 파일의 전체 경로
            wav_file_full_path = os.path.join(input_wav_folder, filename)

            # 파일명에서 확장자를 제거하여 접두사 추출 (예: 'a0001.wav' -> 'a0001')
            filename_prefix = os.path.splitext(filename)[0]

            if filename_prefix in expected_prefixes:
                # 출력 TXT 파일 경로 구성 (예: output_txt_dir/a0001_data.txt)
                output_txt_file_full_path = os.path.join(
                    output_txt_dir, f"{filename_prefix}_data.txt"
                )

                print(f"[{processed_count + 1}] 변환 중: {filename}")
                process_and_save_audio_data(
                    wav_file_full_path, output_txt_file_full_path
                )
                processed_count += 1
            else:
                # REFERENCE.csv에 없는 WAV 파일은 건너뜁니다.
                # print(f"경고: '{filename}'은(는) REFERENCE.csv에 없는 파일 접두사입니다. 건너뜜.")
                skipped_count += 1

        if (processed_count + skipped_count) % 50 == 0 and (
            processed_count + skipped_count
        ) > 0:
            print(f"--- {processed_count + skipped_count} 파일 처리 중 ---")

    print(f"\n--- WAV 파일을 1D TXT로 변환 완료 ---")
    print(
        f"총 {total_expected_files}개의 예상 파일 중 {processed_count}개 파일 변환 성공, {skipped_count}개 파일 건너뜀."
    )


# --- 사용 예시 ---
if __name__ == "__main__":
    # 🚨🚨🚨 실제 데이터셋 경로로 변경해야 합니다 🚨🚨🚨

    # 1. 모든 원본 WAV 파일이 들어있는 단일 폴더 경로
    # 예: 'D:/MDO/heartbeat/all_training_wavs/'
    # ==========================================================경로 지정==========================================================
    input_wav_folder_path = r"D:\MDO\heartbeat/Dataset"

    # 2. 변환된 1D TXT 파일을 저장할 폴더 경로
    # folder_structure.txt에 따르면 [Read_1D_Data]/[result] 안에 있는 것으로 보입니다.
    # ==========================================================경로 지정==========================================================
    output_1d_txt_directory = "Wavelet/Normal_Transformed_Data"

    # 3. 모든 파일명 접두사와 라벨 정보를 담고 있는 REFERENCE.csv 파일 경로
    # 일반적으로 입력 WAV 폴더와 같은 상위 디렉토리에 있습니다.
    # ==========================================================경로 지정==========================================================
    label_csv_file = r"D:\MDO\heartbeat/Dataset/REFERENCE.csv"

    # ❗❗❗ 테스트를 위한 더미 파일 및 폴더 생성 (실제 사용 시에는 이 부분 삭제) ❗❗❗
    # 이 부분은 실제 데이터가 없을 때 코드를 실행해보기 위한 것입니다.
    # 실제 WAV 파일과 REFERENCE.csv가 존재하면 이 블록을 주석 처리하거나 삭제하세요.
    if not os.path.exists(input_wav_folder_path) or not os.listdir(
        input_wav_folder_path
    ):
        print("\n--- 테스트를 위해 더미 WAV 데이터 폴더 및 파일 생성 ---")
        os.makedirs(input_wav_folder_path, exist_ok=True)

        dummy_labels_data = [
            ("a0001", 1),
            ("a0002", 0),
            ("a0003", 1),
            ("a0004", 0),
            ("a0005", 1),
            ("b0001", 1),
            ("b0002", 0),
            ("b0003", 1),
            ("c0001", 1),
            ("c0002", 0),
        ]
        # 더미 REFERENCE.csv 생성
        dummy_df = pd.DataFrame(dummy_labels_data, columns=["filename_prefix", "label"])
        dummy_df.to_csv(label_csv_file, index=False, header=False)
        print(f"더미 '{label_csv_file}' 생성 완료.")

        # 더미 WAV 파일들을 단일 폴더(input_wav_folder_path) 내에 생성
        dummy_sr = 44100  # 샘플링 레이트
        for prefix, _ in dummy_labels_data:
            dummy_wav_path = os.path.join(input_wav_folder_path, f"{prefix}.wav")

            dummy_duration = np.random.randint(2, 6)  # 2~5초 길이의 더미 오디오
            dummy_data = np.random.uniform(
                -0.5, 0.5, int(dummy_sr * dummy_duration)
            ).astype(np.float32)
            try:
                sf.write(dummy_wav_path, dummy_data, dummy_sr)
            except ImportError:
                print(
                    "soundfile 라이브러리가 설치되어 있지 않아 더미 WAV 파일을 생성할 수 없습니다."
                )
                print("pip install soundfile 로 설치해주세요.")
                print(
                    "실제 WAV 파일 경로를 input_wav_folder_path 변수에 직접 지정하여 사용해주세요."
                )
                exit()
        print(
            f"--- 더미 WAV 파일 {len(dummy_labels_data)}개 '{input_wav_folder_path}'에 생성 완료. 이제 변환을 실행할 수 있습니다. ---"
        )
    # ❗❗❗ 더미 파일 생성 끝 ❗❗❗

    # 모든 WAV 파일을 1D TXT로 변환하는 함수 호출
    convert_all_wavs_to_1d_txt(
        input_wav_folder_path, output_1d_txt_directory, label_csv_file
    )
