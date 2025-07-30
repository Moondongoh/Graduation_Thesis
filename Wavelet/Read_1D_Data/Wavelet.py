import librosa
import numpy as np
import pywt
import os


def process_wavelet_and_save_data(
    wav_file_path, output_txt_path, wavelet_name="db1", level=5
):
    """
    1차원 WAV 오디오 파일을 로드하고, 웨이블릿 변환을 적용한 후
    변환된 계수들을 TXT 파일로 저장합니다.

    Args:
        wav_file_path (str): 로드할 WAV 파일의 전체 경로.
        output_txt_path (str): 웨이블릿 변환된 데이터를 저장할 TXT 파일의 전체 경로.
        wavelet_name (str): 사용할 웨이블릿의 이름 (예: 'db1', 'haar', 'sym5' 등).
        level (int): 웨이블릿 분해 레벨.
    """
    try:
        # 1. WAV 파일 로드
        y, sr = librosa.load(wav_file_path, sr=None)
        print(f"'{os.path.basename(wav_file_path)}' 파일 로드 완료.")
        print(f"오디오 데이터 형태: {y.shape}")
        print(f"샘플링 레이트: {sr} Hz")

        # 2. 웨이블릿 변환 적용 (Discrete Wavelet Transform - DWT)
        # coeffs는 (cA_n, cD_n, cD_n-1, ..., cD_1) 형태의 튜플입니다.
        # cA_n: 근사 계수 (저주파 성분)
        # cD_x: 세부 계수 (고주파 성분)
        coeffs = pywt.wavedec(y, wavelet_name, level=level)
        print(f"웨이블릿 변환 ('{wavelet_name}', 레벨 {level}) 적용 완료.")

        # 3. 웨이블릿 변환된 계수들을 하나의 1차원 배열로 평탄화
        # 웨이블릿 계수들은 길이가 다를 수 있으므로, 저장하기 위해 하나로 합칩니다.
        # 이 과정에서 데이터의 원래 구조(각 계수의 의미)는 손실될 수 있으니 주의하십시오.
        # 필요에 따라 각 계수를 개별 파일로 저장하거나 다른 방식으로 처리할 수 있습니다.
        transformed_data = np.concatenate(coeffs)
        print(f"변환된 데이터 형태 (평탄화 후): {transformed_data.shape}")
        print(f"변환된 데이터 (처음 10개 값): {transformed_data[:10]}")

        # 4. 평탄화된 웨이블릿 데이터를 TXT 파일로 저장
        np.savetxt(output_txt_path, transformed_data, fmt="%f", delimiter="\n")
        print(
            f"웨이블릿 변환된 데이터가 '{output_txt_path}'에 성공적으로 저장되었습니다."
        )

    except FileNotFoundError:
        print(f"오류: '{wav_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except ValueError as ve:
        print(f"오류: 웨이블릿 변환 매개변수가 유효하지 않습니다. {ve}")
    except Exception as e:
        print(f"데이터 처리 및 저장 중 오류 발생: {e}")


# --- 사용 예시 ---
if __name__ == "__main__":
    # 이 부분을 사용자의 실제 WAV 파일 경로와 저장할 TXT 파일 경로로 변경해야 합니다.
    # ==========================================================경로 지정==========================================================
    input_wav_file = r"D:\MDO\heartbeat\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a/a0001.wav"
    output_wavelet_txt_file = "a0001_wavelet_data.txt"

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
            exit()

    # 함수 호출: 'db1' 웨이블릿, 레벨 5로 변환하여 저장
    process_wavelet_and_save_data(
        input_wav_file, output_wavelet_txt_file, wavelet_name="db1", level=5
    )
