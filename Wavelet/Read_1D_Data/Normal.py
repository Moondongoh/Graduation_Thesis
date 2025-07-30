import librosa
import numpy as np
import os


def process_and_save_audio_data(wav_file_path, output_txt_path):
    """
    1차원 WAV 오디오 파일을 로드하고, 숫자 데이터를 TXT 파일로 저장합니다.

    Args:
        wav_file_path (str): 로드할 WAV 파일의 전체 경로.
        output_txt_path (str): 숫자 데이터를 저장할 TXT 파일의 전체 경로.
    """
    try:
        # 1. WAV 파일 로드
        # sr=None으로 설정하여 원본 샘플링 레이트 유지
        y, sr = librosa.load(wav_file_path, sr=None)
        print(f"'{os.path.basename(wav_file_path)}' 파일 로드 완료.")
        print(f"오디오 데이터 형태: {y.shape}")
        print(f"샘플링 레이트: {sr} Hz")
        print(f"오디오 데이터 (처음 10개 값): {y[:10]}")

        # 2. 1차원 숫자 데이터를 TXT 파일로 저장
        # fmt='%f'는 부동 소수점 형식으로 저장함을 의미합니다.
        # delimiter='\n'은 각 숫자를 새 줄에 저장함을 의미합니다.
        np.savetxt(output_txt_path, y, fmt="%f", delimiter="\n")
        print(f"1차원 숫자 데이터가 '{output_txt_path}'에 성공적으로 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{wav_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"데이터 처리 및 저장 중 오류 발생: {e}")


# --- 사용 예시 ---
if __name__ == "__main__":
    # 이 부분을 사용자의 실제 WAV 파일 경로와 저장할 TXT 파일 경로로 변경해야 합니다.
    # 예시: PhysioNet Challenge 데이터셋의 'training-a' 폴더 내 'a0001.wav'

    # 예시 WAV 파일 경로 (실제 파일 경로로 변경 필요)
    # 현재 폴더에 sample.wav 파일이 있다고 가정합니다.
    # 또는 'C:/Users/사용자이름/Desktop/training-a/a0001.wav'와 같이 전체 경로를 지정할 수 있습니다.
    # ==========================================================경로 지정==========================================================
    input_wav_file = r"D:\MDO\heartbeat\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a/a0001.wav"

    # 출력 TXT 파일 경로
    output_text_file = "a0001_data.txt"

    # WAV 파일이 없으면 테스트를 위해 더미 WAV 파일을 생성 (실제 사용 시에는 필요 없음)
    if not os.path.exists(input_wav_file):
        print(
            f"경고: '{input_wav_file}' 파일이 존재하지 않습니다. 테스트를 위해 더미 WAV 파일을 생성합니다."
        )
        dummy_sr = 44100  # 샘플링 레이트
        dummy_duration = 5  # 5초 길이
        dummy_data = np.random.uniform(
            -0.5, 0.5, int(dummy_sr * dummy_duration)
        ).astype(np.float32)
        try:
            import soundfile as sf

            sf.write(input_wav_file, dummy_data, dummy_sr)
            print(f"'{input_wav_file}' 더미 파일 생성 완료. 이제 실행할 수 있습니다.")
        except ImportError:
            print(
                "soundfile 라이브러리가 설치되어 있지 않아 더미 WAV 파일을 생성할 수 없습니다."
            )
            print("pip install soundfile 로 설치해주세요.")
            print(
                "실제 WAV 파일 경로를 input_wav_file 변수에 직접 지정하여 사용해주세요."
            )
            exit()  # 더미 파일 생성 실패 시 프로그램 종료

    # 함수 호출
    process_and_save_audio_data(input_wav_file, output_text_file)
