import pandas as pd

# 替换为您的 CSV 文件路径
file_path = './data/data_to_predict.csv'
df = pd.read_csv(file_path, sep=',')

# 计算每个字段非 0 的数量
non_zero_counts = (df != 0).sum()
print(non_zero_counts)
