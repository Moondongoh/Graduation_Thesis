import librosa
import numpy as np
import pywt
import os


def process_all_wav_files_with_wavelet(
    input_dir, output_dir, wavelet_name="db1", level=5
):
    """
    지정된 디렉토리 내의 모든 WAV 파일을 찾아 웨이블릿 변환을 적용하고,
    변환된 계수들을 별도의 출력 디렉토리에 TXT 파일로 저장합니다.

    Args:
        input_dir (str): WAV 파일이 포함된 입력 디렉토리 경로.
        output_dir (str): 웨이블릿 변환된 데이터를 저장할 출력 디렉토리 경로.
        wavelet_name (str): 사용할 웨이블릿의 이름 (예: 'db1', 'haar', 'sym5' 등).
        level (int): 웨이블릿 분해 레벨.
    """
    if not os.path.exists(input_dir):
        print(
            f"오류: 입력 디렉토리 '{input_dir}'를 찾을 수 없습니다. 경로를 확인해주세요."
        )
        return

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 '{output_dir}'를 생성했습니다.")

    processed_count = 0
    skipped_count = 0

    print(f"\n'{input_dir}' 디렉토리에서 WAV 파일 처리를 시작합니다...")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            wav_file_path = os.path.join(input_dir, filename)
            # 출력 파일명은 원본 WAV 파일명에서 확장자를 .txt로 변경
            output_txt_filename = os.path.splitext(filename)[0] + "_wavelet.txt"
            output_txt_path = os.path.join(output_dir, output_txt_filename)

            print(f"\n--- '{filename}' 처리 중 ---")
            try:
                # 1. WAV 파일 로드
                y, sr = librosa.load(wav_file_path, sr=None)
                print(
                    f"  파일 로드 완료. 오디오 데이터 형태: {y.shape}, 샘플링 레이트: {sr} Hz"
                )

                # 2. 웨이블릿 변환 적용
                coeffs = pywt.wavedec(y, wavelet_name, level=level)
                print(f"  웨이블릿 변환 ('{wavelet_name}', 레벨 {level}) 적용 완료.")

                # 3. 웨이블릿 변환된 계수들을 하나의 1차원 배열로 평탄화
                transformed_data = np.concatenate(coeffs)
                print(f"  변환된 데이터 형태 (평탄화 후): {transformed_data.shape}")

                # 4. 평탄화된 웨이블릿 데이터를 TXT 파일로 저장
                np.savetxt(output_txt_path, transformed_data, fmt="%f", delimiter="\n")
                print(
                    f"  웨이블릿 변환된 데이터가 '{output_txt_path}'에 성공적으로 저장되었습니다."
                )
                processed_count += 1

            except Exception as e:
                print(f"  오류: '{filename}' 처리 중 문제 발생 - {e}")
                skipped_count += 1
        else:
            # .wav 파일이 아닌 경우 건너뜀
            print(f"--- '{filename}' 건너뜀 (WAV 파일이 아님) ---")
            skipped_count += 1

    print(f"\n--- 처리 완료 ---")
    print(f"총 {processed_count}개 파일 처리됨.")
    print(f"총 {skipped_count}개 파일 건너뜀.")


# --- 사용 예시 ---
if __name__ == "__main__":
    # 이 부분을 실제 WAV 파일들이 모여있는 "Dataset" 폴더 경로로 변경해야 합니다.
    # 예시: 'C:/Users/사용자이름/Desktop/Dataset'
    # 현재 스크립트와 같은 위치에 'Dataset' 폴더가 있다고 가정합니다.
    # ==========================================================경로 지정==========================================================
    input_directory = r"D:\MDO\heartbeat/Dataset"

    # 웨이블릿 변환 결과를 저장할 폴더 경로
    # ==========================================================경로 지정==========================================================
    output_directory = r"D:\MDO\heartbeat/Wavelet/Convert/Wavelet_Transformed_Data"

    # 테스트를 위해 더미 'Dataset' 폴더와 WAV 파일을 생성 (실제 사용 시에는 필요 없음)
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
        print(f"'{input_directory}' 더미 폴더 생성 완료.")
        # 더미 WAV 파일 3개 생성
        dummy_sr = 44100
        for i in range(1, 4):
            dummy_wav_file = os.path.join(input_directory, f"dummy_audio_{i}.wav")
            dummy_duration = 2 + i  # 파일마다 길이 다르게
            dummy_data = np.random.uniform(
                -0.5, 0.5, int(dummy_sr * dummy_duration)
            ).astype(np.float32)
            try:
                import soundfile as sf

                sf.write(dummy_wav_file, dummy_data, dummy_sr)
                print(f"'{dummy_wav_file}' 더미 파일 생성 완료.")
            except ImportError:
                print(
                    "soundfile 라이브러리가 설치되어 있지 않아 더미 WAV 파일을 생성할 수 없습니다."
                )
                print("pip install soundfile 로 설치해주세요.")
                print(
                    "실제 WAV 파일 경로를 input_directory 변수에 직접 지정하여 사용해주세요."
                )
                exit()  # 더미 파일 생성 실패 시 프로그램 종료

    # 함수 호출
    # 'db4' 웨이블릿, 레벨 4로 변환 예시
    process_all_wav_files_with_wavelet(
        input_directory, output_directory, wavelet_name="db4", level=4
    )
