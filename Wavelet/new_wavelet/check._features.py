import numpy as np

# 예시 파일 경로
file_path = r"D:\MDO\heartbeat\Features\a0058.npy"

# 로드
features = np.load(file_path)

# 정보 출력
print(f"✅ 특징 벡터 shape: {features.shape}")  # 예: (80, 30)
print(f"🔍 첫 번째 세그먼트 벡터 (길이={len(features[0])}):\n{features[0]}")
