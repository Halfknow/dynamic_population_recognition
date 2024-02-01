# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump, load
from sklearn.pipeline import Pipeline


# 加载数据的函数，输入文件路径，返回pandas数据帧
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess_data(data: pd.DataFrame, target: str, categorical_features: list, numerical_features: list, preprocessor=None) -> tuple:
    # 如果没有提供预处理器，则创建一个新的
    if preprocessor is None:
        # 分类特征的转换器
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # 预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', 'passthrough', numerical_features)
            ])

        # 拟合预处理器并转换数据
        X = preprocessor.fit_transform(data[categorical_features + numerical_features])
        dump(preprocessor, 'preprocessor.joblib')  # 保存预处理器
    else:
        # 使用提供的预处理器转换数据
        X = preprocessor.transform(data[categorical_features + numerical_features])

    # 标签编码
    label_encoder = LabelEncoder()

    # 这里我们希望"否"为0，"是"为1
    values = ['否', '是']
    # 训练编码器
    label_encoder.fit(values)

    y = label_encoder.fit_transform(data[target])

    return X, y


# 划分数据为训练集和测试集的函数
# 输入：特征矩阵X和目标向量y
# 输出：划分后的训练集和测试集的特征矩阵和目标向量
def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    # 划分数据集，其中20%为测试集，随机种子 random_state 设为42（确保每次划分都能得到相同的结果）
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 新增加载预处理器的函数
def load_preprocessor(path: str):
    return load(path)