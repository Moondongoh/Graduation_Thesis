import pandas as pd


def add_labels_to_features(feature_csv_path, reference_csv_path, output_csv_path):
    """
    특징 데이터 CSV에 라벨 정보를 추가하고 새로운 CSV로 저장하는 함수입니다.

    Args:
        feature_csv_path (str): 'file' 열이 포함된 특징 CSV 파일 경로.
        reference_csv_path (str): 파일명과 라벨이 포함된 참조 CSV 파일 경로.
        output_csv_path (str): 라벨이 추가된 최종 CSV 파일 저장 경로.
    """
    try:
        # 1. 특징 데이터 불러오기
        # 'file' 열이 마지막에 있다고 가정하고 읽습니다.
        df_features = pd.read_csv(feature_csv_path)
        print(f"✅ 특징 데이터 로드 완료: {df_features.shape}")

        # 2. 참조 라벨 데이터 불러오기
        # 'REFERENCE.csv' 파일은 헤더가 없으므로 header=None으로 지정하고,
        # 컬럼 이름을 'filename'과 'label'로 지정합니다.
        df_labels = pd.read_csv(
            reference_csv_path, header=None, names=["filename", "label"]
        )
        print(f"✅ 참조 라벨 로드 완료: {df_labels.shape}")

        # 3. 파일 확장자 제거
        # 두 데이터프레임의 파일명 형식을 통일합니다.
        df_features["file"] = df_features["file"].str.replace(".wav", "", regex=False)
        df_labels["filename"] = df_labels["filename"].str.replace(
            ".wav", "", regex=False
        )

        # 4. 데이터 병합 (Merge)
        # 'filename'을 기준으로 두 데이터를 합칩니다.
        # on='filename'은 병합의 기준이 되는 열을 의미합니다.
        final_df = pd.merge(
            df_features, df_labels, left_on="file", right_on="filename", how="inner"
        )
        print(f"✅ 데이터 병합 완료: {final_df.shape}")

        # 5. 불필요한 열 삭제
        # 병합 후 중복되는 'filename' 열과 원본 'file' 열을 삭제합니다.
        final_df = final_df.drop(columns=["file", "filename"])

        # 6. 최종 데이터를 새로운 CSV 파일로 저장
        final_df.to_csv(output_csv_path, index=False)
        print(f"🎉 최종 데이터셋 저장 완료: '{output_csv_path}'")

    except FileNotFoundError as e:
        print(f"❌ 오류: 파일이 존재하지 않습니다. 경로를 다시 확인해주세요: {e}")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")


FEATURE_CSV_PATH = r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_validation.csv"  # 웨이블릿 특징 CSV
REFERENCE_CSV_PATH = (
    r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\validation\REFERENCE2.csv"  # 원본 라벨 CSV
)
OUTPUT_CSV_PATH = (
    r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\wavelet_validation.csv"  # 최종 결과 CSV
)

add_labels_to_features(FEATURE_CSV_PATH, REFERENCE_CSV_PATH, OUTPUT_CSV_PATH)
