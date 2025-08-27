import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def visualize_wav_file(input_path, output_path):
    try:
        # 오디오 파일 읽기
        samplerate, data = wavfile.read(input_path)

        # 데이터가 비어 있는지 확인
        if data.size == 0:
            print(f"경고: {os.path.basename(input_path)} 파일이 비어있어 건너뜁니다.")
            return

        # 페어링을 위해 데이터 샘플 수를 짝수로 맞춤
        if len(data) % 2 != 0:
            data = data[:-1]

        # x, y 좌표 생성
        x = data[0::2]
        y = data[1::2]

        # 스캐터 플롯 생성
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5, s=5)
        plt.title(f"Audio Visualization: {os.path.basename(input_path)}")
        plt.xlabel("Sample (2n)")
        plt.ylabel("Sample (2n+1)")
        plt.grid(True)

        # 이미지 파일로 저장
        plt.savefig(output_path)
        plt.close()  # 메모리 해제를 위해 플롯 닫기
        print(
            f"성공: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"
        )

    except Exception as e:
        print(f"오류: {os.path.basename(input_path)} 처리 중 오류 발생 - {e}")


def process_folder(input_folder, output_folder):
    # 결과물을 저장할 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    # 입력 폴더의 모든 파일 목록 가져오기
    for filename in os.listdir(input_folder):
        # .wav 파일인지 확인
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_folder, filename)

            # 출력 파일 이름 설정 (확장자 변경)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)

            # 시각화 함수 호출
            visualize_wav_file(input_path, output_path)


# --- 여기를 수정하세요 ---
# WAV 파일이 있는 폴더 경로
input_folder = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\train"
# PNG 이미지를 저장할 폴더 이름
output_folder = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\wav_visualizations"
# -------------------------

# 메인 함수 실행
if __name__ == "__main__":
    process_folder(input_folder, output_folder)
