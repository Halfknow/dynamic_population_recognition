import pandas as pd

# 替换为您的 CSV 文件路径
file_path = './data/data_to_predict.csv'
df = pd.read_csv(file_path, sep=',')

# 计算每个字段非 0 的数量
non_zero_counts = (df != 0).sum()
print(non_zero_counts)

non_zero_df = df[(df['YI_NIAN_YI_YUAN'] != 0) | (df['DANG_DI_JIAO_SHI'] != 0) | (df['DANG_DI_XUE_SHENG'] != 0) | (df['GONG_ZU_FANG'] != 0) | (df['SHE_HUI_JIU_ZHU'] != 0) | (df['BEN_DI_FA_REN'] != 0) | (df['SANSHI_TIAN_TING_CHE'] != 0) | (df['BAN_NIAN_TING_CHE'] != 0) | (df['SANSHI_TIAN_GONG_JIAO'] != 0) | (df['BAN_NIAN_GONG_JIAO'] != 0)]


# 保存到 CSV 文件
non_zero_df.to_csv('./data/data_to_predict_final.csv', index=False)
print(non_zero_df)