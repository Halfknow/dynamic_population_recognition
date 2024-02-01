import pandas as pd

# 替换为您的 CSV 文件路径
file_path = './data/data_predicted.csv'
df = pd.read_csv(file_path, sep=',')

# 计算判断结果字段为否的数量
pan_duan_fou = df[df['PAN_DUAN_JIE_GUO'] != 0]
print(pan_duan_fou)
print(len(df))
print(len(pan_duan_fou))
