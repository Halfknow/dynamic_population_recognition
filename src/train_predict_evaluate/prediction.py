# prediction.py
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from numpy import ndarray

def make_prediction(model: BaseEstimator, X: csr_matrix) -> ndarray:
    """
    使用训练好的模型进行预测
    :param model: 训练好的模型，类型为 sklearn.base.BaseEstimator
    :param X: 需要进行预测的特征矩阵，类型为 scipy.sparse.csr_matrix
    :return: 预测结果，类型为 numpy.ndarray
    """
    return model.predict(X)
