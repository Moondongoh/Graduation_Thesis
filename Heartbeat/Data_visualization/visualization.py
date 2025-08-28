import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def find_global_max_val(input_folder):
    global_max = 0
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]

    if not file_list:
        print(f"경고: '{input_folder}' 폴더에 WAV 파일이 없습니다.")
        return 0

    print("--- 1. 전체 WAV 파일의 최대값 탐색 시작 ---")
    for filename in file_list:
        input_path = os.path.join(input_folder, filename)
        try:
            # 오디오 파일 읽기
            samplerate, data = wavfile.read(input_path)
            if data.size > 0:
                current_max = np.max(np.abs(data))
                if current_max > global_max:
                    global_max = current_max
                    print(f"새로운 최대값 발견: {global_max:.2f} (파일: {filename})")
        except Exception as e:
            print(f"경고: '{filename}' 처리 중 오류 발생 - {e}")

    print(f"\n✅ 전체 WAV 파일의 최종 최대값: {global_max:.2f}")
    return global_max


def visualize_wav_file(input_path, output_path, global_max_val):
    try:
        samplerate, data = wavfile.read(input_path)

        if data.size == 0:
            print(f"경고: {os.path.basename(input_path)} 파일이 비어있어 건너뜁니다.")
            return

        if len(data) % 2 != 0:
            data = data[:-1]

        x = data[0::2]
        y = data[1::2]

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, alpha=0.5, s=5)

        limit = global_max_val * 1.05
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

        plt.title(f"Audio Visualization: {os.path.basename(input_path)}")
        plt.xlabel("Sample (2n)")
        plt.ylabel("Sample (2n+1)")
        plt.grid(True)

        plt.savefig(output_path)
        plt.close()
        print(
            f"성공: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"
        )

    except Exception as e:
        print(f"오류: {os.path.basename(input_path)} 처리 중 오류 발생 - {e}")


def process_folder(input_folder, output_folder):
    """폴더의 모든 WAV 파일을 처리합니다."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    global_max_val = find_global_max_val(input_folder)

    if global_max_val == 0:
        print("최대값이 0이므로 이미지 생성을 중단합니다.")
        return

    print("\n--- 2. PNG 이미지 생성 시작 ---")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            visualize_wav_file(input_path, output_path, global_max_val)


input_folder = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\train"
output_folder = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\new_wav_visualizations"

if __name__ == "__main__":
    process_folder(input_folder, output_folder)
