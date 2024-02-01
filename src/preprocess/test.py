import pandas as pd

# 替换为您的 CSV 文件路径
file_path = './data/DWD_NBBL_RKTZ_DTRKTZ_DF_202401291630.csv'
df = pd.read_csv(file_path, sep=',')
print("df", (df['SHE_QU_PAI_CHA_ZZ'].notna()).sum())