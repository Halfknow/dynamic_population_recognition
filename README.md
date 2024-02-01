# Dynamic Population Recognition
北仑区动态人口预测，基于scikit-learn的逻辑回归机器学习算法，对北仑区人口数据进行二分类预测，预测结果通过Shap库进行可解释性分析。

## Requirements
```
Python>=3.11
scikit-learn>=1.3.2
pandas>=2.0.3
numpy>=1.24.4
scipy>=1.9.3
shap>=0.44.0
```

## Usage
### Data preparation
#### Data source
动态人口预测处理过程中涉及的数据表如下所示：
| 序号 | 表名| 文件名 | 表说明 |
| :---: | :---: | :---: | :---: |
| 1 | 人口数据表 | DWD_NBBL_RKTZ_DTRKTZ_DF_202311211623.csv | 北仑区人口数据（数据量400万） | 
| 2 | 人口决策树表 | data_decided.csv | 北仑区人口数据，经过决策树打标（数据量230万） |
| 3 | 人口预测表 | data_to_predict.csv | 需要预测的人口表（数据量170万） |
| 4 | 人口预测结果表 | data_to_predict_final.csv | 最终需要预测的人口结果表（数据量3万） |
| 5 | 人口预测结果表 | data_predicted.csv | 预测结果表（数据量3万） |
| 6 | 人口预测结果融合表 | DWD_NBBL_RKTZ_DTRKTZ_DF_UPDATED_202312151153.csv | 北仑动态人口预测结果表（数据量400万） |

### Training

### Evaluation

### Prediction

### Visualization

### 文件执行顺序
```
data_preprocessing.py - 数据预处理
data_reader.py - 数据读取
data_to_predict_final.py - 最终预测数据处理
data_to_predict_statistics.py - 预测数据统计
main.py - 主程序
model_evaluation.py - 模型评估
model_training.py - 模型训练
prediction.py - 预测
data_decided_statistics.py - 已决策数据统计
data_predicted_statistics.py - 已预测数据统计
```
