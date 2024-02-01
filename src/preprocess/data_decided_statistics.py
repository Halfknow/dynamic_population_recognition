import pandas as pd

# 替换为您的 CSV 文件路径
file_path = './data/data_decided.csv'
df = pd.read_csv(file_path, sep=',')

# 计算判断结果字段为是的数量
pan_duan_shi = df[df['PAN_DUAN_JIE_GUO'] == "是"]
print(pan_duan_shi)
print(len(pan_duan_shi))