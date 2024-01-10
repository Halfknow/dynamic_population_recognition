# 1st step
import pandas as pd

# 替换为您的 CSV 文件路径
file_path = './data/DWD_NBBL_RKTZ_DTRKTZ_DF_202311211623.csv'
df = pd.read_csv(file_path, sep=',')
print("df", df)
# 打印每个字段的数据类型
print(df.dtypes)

# 打印每个字段的 NaN 数量
nan_counts = df.isna().sum()
print(nan_counts)
# "ZJ","SAN_YUE_YI_YUAN","YI_NIAN_YI_YUAN","DANG_DI_JIAO_SHI","DANG_DI_XUE_SHENG","DANG_DI_ZAI_ZHI","GONG_JI_JIN","BU_DONG_CHAN_SHU_LIANG","GONG_ZU_FANG","JIN_QI_JIAO_YI","SHE_HUI_JIU_ZHU","BEN_DI_FA_REN","SANSHI_TIAN_TING_CHE","BAN_NIAN_TING_CHE","SANSHI_TIAN_GONG_JIAO","BAN_NIAN_GONG_JIAO","PAN_DUAN_JIE_GUO","PAN_DUAN_LI_YOU","SI_WANG_ZHENG_MING","SAN_TIAN_SAN_JIAN","HU_JI_REN_KOU_ZC","JIAO_NA_SHE_BAO","ZAI_XIAO_XUE_SHENG","HU_JI_REN_KOU_ZX","LIU_DONG_REN_KOU_ZX","JU_ZHU_ZHENG_ZX"

# 指定字段列表
fields = ["SANSHI_TIAN_TING_CHE", "BAN_NIAN_TING_CHE", "SANSHI_TIAN_GONG_JIAO", "BAN_NIAN_GONG_JIAO"]

# 将这些字段的 NaN 值填充为 0
df[fields] = df[fields].fillna(0)
df[fields] = df[fields].astype('int64')

# 用于预测的数据（PAN_DUAN_JIE_GUO 为空），用于训练的数据（PAN_DUAN_JIE_GUO 不为空）
data_predict = df[df['PAN_DUAN_JIE_GUO'].isna()]
data_for_train = df[df['PAN_DUAN_JIE_GUO'].notna()]

# 保存到 CSV 文件
data_predict.to_csv('./data/data_to_predict.csv', index=False)
data_for_train.to_csv('./data/data_decided.csv', index=False)
