"""
라벨 값 -1을 0으로 바꾸기
"""

import pandas as pd

df = pd.read_csv("REFERENCE.csv")

df_replaced = df.replace(-1, 0)

df_replaced.to_csv("REFERENCE2.csv", index=False)
