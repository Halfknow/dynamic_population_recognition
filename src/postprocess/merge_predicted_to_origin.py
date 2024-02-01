import pandas as pd
import time

# 读取原始数据和预测数据
original_data_path = "./data/DWD_NBBL_RKTZ_DTRKTZ_DF_202311211623.csv"
predicted_data_path = "./data/data_predicted.csv"
original_data = pd.read_csv(original_data_path)
predicted_data = pd.read_csv(predicted_data_path)

# 确保ZJ列的数据类型一致
original_data['ZJ'] = original_data['ZJ'].astype(str)
predicted_data['ZJ'] = predicted_data['ZJ'].astype(str)

# 使用ZJ作为主键合并数据，只更新原始数据中PAN_DUAN_JIE_GUO为NaN的行
original_data = original_data.set_index('ZJ')
predicted_data = predicted_data.set_index('ZJ')
original_data.update(predicted_data)

current_time = time.strftime("%Y%m%d%H%M")

# 重置索引并保存结果
original_data.reset_index(inplace=True)
original_data.to_csv(f"./data/DWD_NBBL_RKTZ_DTRKTZ_DF_UPDATED_{current_time}.csv", index=False)