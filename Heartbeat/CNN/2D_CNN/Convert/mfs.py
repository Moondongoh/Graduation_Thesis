import os
import librosa
import numpy as np

def extract_features(file_path, num_mfcc=13, num_mels=40):
    """
    오디오 파일에서 MFCC와 Melspectrogram(MFS) 특징을 추출하는 함수입니다.
    """
    try:
        # 오디오 파일 로드
        y, sr = librosa.load(file_path, sr=None)

        # MFCC 추출
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)

        # Melspectrogram(MFS) 추출
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels)

        # 로그 스케일로 변환하여 2D 데이터로 만듭니다.
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # MFCC와 Melspectrogram을 2차원 배열로 결합합니다.
        # 동일한 프레임 수를 가지도록 전치하고 결합합니다.
        combined_features = np.vstack([mfccs, log_melspec])

        return combined_features

    except Exception as e:
        print(f"오류 발생: {file_path}, 오류 내용: {e}")
        return None

def process_dataset(input_dir, output_dir):
    """
    지정된 폴더의 모든 오디오 파일에 대해 특징을 추출하고 저장하는 함수입니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 폴더를 생성했습니다.")

    processed_count = 0
    # 입력 폴더가 존재하는지 확인
    if not os.path.exists(input_dir):
        print(f"오류: 입력 폴더 '{input_dir}'를 찾을 수 없습니다.")
        return
        
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)

            features = extract_features(file_path)

            if features is not None:
                # 출력 파일 경로 설정
                base_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(output_dir, f"{base_name}.npy")

                # NumPy 배열로 저장
                np.save(output_path, features)
                processed_count += 1
                print(f"'{file_name}'를 성공적으로 변환하여 '{output_path}'에 저장했습니다.")

    print(f"\n총 {processed_count}개의 파일을 처리했습니다.")

# 실행
# input_folder = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv/Dataset'
# output_folder = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv/processed_data'
input_folder = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv/validation'
output_folder = r'D:\MDO\heartbeat\1_New_HB_0818\change_csv/validation_data'
process_dataset(input_folder, output_folder)