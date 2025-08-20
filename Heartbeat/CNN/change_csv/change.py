import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('REFERENCE.csv')

# 데이터프레임의 모든 '-1' 값을 '0'으로 변경
df_replaced = df.replace(-1, 0)

# 수정된 데이터프레임을 새로운 CSV 파일로 저장
df_replaced.to_csv('REFERENCE2.csv', index=False)