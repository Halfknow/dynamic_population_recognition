# main.py
import data_preprocessing as dp
import model_training as mt
import model_evaluation as me
import os
import matplotlib.pyplot as plt
import shap
import numpy as np


# 设定数据存储路径
DIR_PATH = os.path.expanduser('~') + '/Desktop/dynamic_population_recognition/'
DESKTOP_PATH = os.path.expanduser('~') + '/Desktop/'

# 假设数据文件路径已定义
data_file = DIR_PATH + 'data/data_decided.csv'
new_data_file = DIR_PATH + 'data/data_to_predict_final.csv'

# 数据预处理
data = dp.load_data(data_file)
target = 'PAN_DUAN_JIE_GUO'
target_predict = 'PAN_DUAN_JIE_GUO_ML'

categorical_features = ['SAN_YUE_YI_YUAN','YI_NIAN_YI_YUAN','DANG_DI_JIAO_SHI','DANG_DI_XUE_SHENG','DANG_DI_ZAI_ZHI','GONG_JI_JIN','BU_DONG_CHAN_SHU_LIANG','GONG_ZU_FANG','JIN_QI_JIAO_YI','SHE_HUI_JIU_ZHU','BEN_DI_FA_REN','SI_WANG_ZHENG_MING','HU_JI_REN_KOU_ZC','JIAO_NA_SHE_BAO','ZAI_XIAO_XUE_SHENG','HU_JI_REN_KOU_ZX','LIU_DONG_REN_KOU_ZX','JU_ZHU_ZHENG_ZX','SHE_QU_PAI_CHA_ZZ']  # 分类特征
numerical_features = ['SANSHI_TIAN_TING_CHE','BAN_NIAN_TING_CHE','SANSHI_TIAN_GONG_JIAO','BAN_NIAN_GONG_JIAO']  # 数值特征

print("categories features length: ", len(categorical_features))
print("numerical features length: ", len(numerical_features))

X, y = dp.preprocess_data(data, target, categorical_features, numerical_features)
X_train, X_test, y_train, y_test = dp.split_data(X, y)

# 训练模型
model = mt.train_model(X_train, y_train)

# 评估模型
me.evaluate_model(model, X_test, y_test)

# 加载预处理器
preprocessor = dp.load_preprocessor('preprocessor.joblib')

# 假设 preprocessor 是一个 ColumnTransformer
feature_names = preprocessor.get_feature_names_out()

# 假设 model 是你的逻辑回归模型
coefficients = model.coef_[0]  # 获取模型系数

# 确保特征名称和系数长度相同
assert len(coefficients) == len(feature_names)

# 创建一个特征名称和系数的字典
feature_coefficient_dict = dict(zip(feature_names, coefficients))

# 打印每个特征及其对应的系数
for feature, coef in feature_coefficient_dict.items():
    print(f"Feature: {feature}, Coefficient: {coef}")

# 打印模型权重和截距
print("Model coefficients: ", model.coef_, len(model.coef_[0]))
print("Model intercept: ", model.intercept_)

# 对新数据进行预测
new_data = dp.load_data(new_data_file)
new_data_processed, _ = dp.preprocess_data(new_data, target_predict, categorical_features, numerical_features, preprocessor)
predictions = model.predict(new_data_processed)

print("predictions: ", predictions)
print("predictions_type: ", type(predictions))

# 将预测结果转换为DataFrame
new_data[target_predict] = predictions

# 保存预测好的数据
new_data.to_csv('./data/data_predicted.csv', index=False)

#################################### 使用 SHAP 进行可视化 ##########################################
# 创建一个解释器对象
explainer = shap.Explainer(model, X_train)
shap_values = explainer.shap_values(X_test)  # 计算SHAP值

# 获取全部特征名称
all_feature_names = preprocessor.get_feature_names_out(categorical_features + numerical_features)

# 过滤出不以 '_0' 结尾的特征名称
filtered_feature_names = [name for name in all_feature_names if not name.endswith('_0')]
# 获取过滤后特征名称的索引
filtered_indices = [i for i, name in enumerate(all_feature_names) if not name.endswith('_0')]

# 对 SHAP 值应用过滤
shap_values_filtered = shap_values[:, filtered_indices]

# 现在过滤 X_test 数据
X_test_filtered = X_test[:, filtered_indices]

# 计算每个特征的平均绝对SHAP值
mean_abs_shap = np.abs(shap_values_filtered).mean(axis=0)

# 将特征名称和它们的平均绝对SHAP值结合在一起
feature_importance = list(zip(filtered_feature_names, mean_abs_shap))

# 根据SHAP值降序排序
feature_importance.sort(key=lambda x: x[1], reverse=True)

# 打印特征影响力降序列表
for feature, importance in feature_importance:
    print(f"Feature: {feature}, SHAP Value: {importance}")

# SHAP摘要图 (Summary Plot) 
plt.figure()
# 使用过滤后的特征名称列表
# shap.summary_plot(shap_values, X_test, feature_names=all_feature_names, show = False)
shap.summary_plot(shap_values_filtered, X_test_filtered, feature_names=filtered_feature_names, show = False)
plt.savefig(DIR_PATH + 'shap_summary_1.png')
# 如需在屏幕上显示图像，可在保存后调用

# # 力量图 (Force Plot) 选择一个样本进行可视化
# sample_index = 0  # 可以选择不同的索引
# shap_html = shap.force_plot(explainer.expected_value, shap_values_filtered[sample_index, :], X_test_filtered[sample_index, :], feature_names=filtered_feature_names)
# shap.save_html(DIR_PATH + 'shap_force_plot_sample_' + str(sample_index) + '.html', shap_html)

# 依赖图 (Dependence Plot) 选择一个特征进行可视化
# plt.figure()
# feature_to_plot = filtered_feature_names[0]  # 替换为实际的特征名称
# shap.dependence_plot(feature_to_plot, shap_values_filtered, X_test_filtered, feature_names=filtered_feature_names)
# plt.savefig(DIR_PATH + 'shap_dependence_plot_' + feature_to_plot + '.png')

# 决策图 (Decision Plot) 为多个样本绘制决策图
# 选择一个样本子集进行绘图
plt.figure()
sample_indices = range(10)  # 例如，选择前 10 个样本
shap_values_filtered_subset = shap_values_filtered[sample_indices]
X_test_filtered_subset = X_test_filtered[sample_indices]
# shap.decision_plot(explainer.expected_value, shap_values, X_test, feature_names=all_feature_names, feature_display_range=slice(-1, -20, -1), show = False)
shap.decision_plot(explainer.expected_value, shap_values_filtered_subset, X_test_filtered_subset, feature_names=filtered_feature_names, feature_display_range=slice(-1, -20, -1), show=False)
plt.savefig(DIR_PATH + 'shap_decision_plot_20_samples_1.png', bbox_inches='tight')

# 如需在屏幕上显示图像，可在保存后调用
plt.show()