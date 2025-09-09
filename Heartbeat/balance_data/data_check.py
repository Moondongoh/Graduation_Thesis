'''
현재 데이터셋은 불균형 상태 1 = 비정상, -1 = 정상
정상 데이터가 4배 정도 더 많은 상태이다.

학습을 위해서 -1(음성 데이터 = 정상)을 0으로 변경

그리고 Physical Copy와 SMOTE 기법을 통해 오버 샘플링, 그리고 GAN을 이용해 불균형을 해소할 예정이다.
'''

import pandas as pd
import matplotlib.pyplot as plt

# csv_path = r'D:\MDO\heartbeat\1_New_HB_0818\Dataset\train\REFERENCE2.csv'
# csv_path = r'D:\MDO\heartbeat\1_New_HB_0818\balance_data\Physical_copy\Dataset\REFERENCE2_balanced.csv'
csv_path = r'D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Dataset\REFERENCE2_gan_balanced.csv'

df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])

# df['label'] = df['label'].map({-1: 0, 1: 1})

label_counts = df['label'].value_counts().sort_index()

plt.figure(figsize=(5, 4))
label_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('dataset')
# plt.xlabel('label (-1: Normal, 1: Abnormal)')
plt.ylabel('count')
plt.xticks([0, 1], ['Normal(-1)', 'Abnormal(1)'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()