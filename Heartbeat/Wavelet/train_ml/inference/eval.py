import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

def run_inference_from_csv(model_path, data_folder_path, data_file_name, label_column='label'):
    """
    미리 전처리된 CSV 데이터를 사용하여 모델 추론 및 평가를 수행합니다.

    Args:
        model_path (str): 저장된 모델 파일 경로 (예: 'random_forest_model.joblib').
        data_folder_path (str): 전처리된 CSV 파일이 있는 폴더 경로.
        data_file_name (str): 전처리된 CSV 파일명 (예: 'wavelet_validation.csv').
        label_column (str): 데이터프레임에서 라벨을 포함하는 열의 이름.
    """
    
    # 1. 학습된 모델 로드
    try:
        model = joblib.load(model_path)
        print(f"✅ 모델 로드 완료: '{model_path}'")
    except FileNotFoundError:
        print(f"❌ 오류: '{model_path}' 파일을 찾을 수 없습니다. 모델 학습 코드를 먼저 실행하여 저장했는지 확인해주세요.")
        return
        
    # 2. 전처리된 데이터 로드
    file_path = os.path.join(data_folder_path, data_file_name)
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로드 완료: {df.shape}")
    except FileNotFoundError:
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다. 전처리 과정이 완료되었는지 확인해주세요.")
        return
    
    # 3. 특징(X)과 라벨(y) 분리
    if label_column not in df.columns:
        print(f"❌ 오류: 데이터프레임에 라벨 컬럼('{label_column}')이 없습니다.")
        return
        
    X_val = df.drop(label_column, axis=1)
    y_val = df[label_column]

    print(f"\n✅ 검증 데이터 준비 완료. 샘플 수: {len(X_val)}")
    
    # 4. 추론 및 평가
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    print("\n--- 검증 데이터 추론 및 성능 평가 ---")
    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print(f"균형 정확도 (Balanced Accuracy): {bal_acc:.4f}")
    print("\n분류 리포트:")
    print(report)


# --- 사용 예시 ---
if __name__ == '__main__':
    MODEL_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\Wavelet/model\random_forest_model.joblib'
    
    # 전처리가 완료된 CSV 파일이 있는 폴더
    DATA_FOLDER_PATH = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset' 
    
    # 해당 폴더 내의 CSV 파일명
    DATA_FILE_NAME = 'wavelet_validation.csv'
    
    run_inference_from_csv(MODEL_PATH, DATA_FOLDER_PATH, DATA_FILE_NAME)