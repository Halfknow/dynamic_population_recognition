from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from pandas import Series
from sklearn.base import BaseEstimator

def train_model(X_train: csr_matrix, y_train: Series) -> BaseEstimator:
    """
    训练逻辑回归模型的函数
    :param X_train: 训练集的特征矩阵，类型为 scipy.sparse.csr_matrix
    :param y_train: 训练集的目标向量，类型为 pandas.Series
    :return: 训练好的模型，类型为 sklearn.base.BaseEstimator
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model
