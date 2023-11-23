# model_evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from pandas import Series
import numpy as np
import os
import datetime

# 获取当前时间
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%y%m%d%H%M%S") # 格式化时间为 'YYMMDDHHMMSS' 格式


# 设定数据存储路径
DIR_PATH = os.path.expanduser('~') + '/Desktop/dynamic_population_recognition/'
DESKTOP_PATH = os.path.expanduser('~') + '/Desktop/'

def evaluate_model(model: BaseEstimator, X_test: csr_matrix, y_test: Series) -> None:
    """
    评估机器学习模型的性能
    :param model: 训练好的模型，类型为 sklearn.base.BaseEstimator
    :param X_test: 测试集的特征矩阵，类型为 scipy.sparse.csr_matrix
    :param y_test: 测试集的目标向量，类型为 pandas.Series
    :return: None，函数直接打印出模型的评估结果
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 构建带有时间戳的文件名
    filename = DIR_PATH + "evaluation_results/evaluation_" + formatted_time + ".txt"

    with open(filename, "w") as file:
        file.write("Classification Report:\n")
        file.write(classification_report(y_test, y_pred))
        file.write("\nAccuracy: " + str(accuracy_score(y_test, y_pred)) + "\n")
        file.write("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)) + "\n")
        file.write("\nROC Curve Area: " + str(roc_auc) + "\n")

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
