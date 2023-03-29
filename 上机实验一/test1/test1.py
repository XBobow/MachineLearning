from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# 导入数据集
#`load_boston` has been removed from scikit-learn since version 1.2.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 画散点图
# Variables in order:
#  CRIM     per capita crime rate by town
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  INDUS    proportion of non-retail business acres per town
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  NOX      nitric oxides concentration (parts per 10 million)
#  RM       average number of rooms per dwelling
#  AGE      proportion of owner-occupied units built prior to 1940
#  DIS      weighted distances to five Boston employment centres
#  RAD      index of accessibility to radial highways
#  TAX      full-value property-tax rate per $10,000
#  PTRATIO  pupil-teacher ratio by town
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  LSTAT    % lower status of the population
#  MEDV     Median value of owner-occupied homes in $1000's
feature = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"] #MEDV是价值
# for i in range(13):
#     plt.figure(figsize=(10, 7))
#     plt.grid()
#     plt.scatter(data[:, i], target, s=5)
#     plt.title(feature[i])
# plt.show()

# 数据归一化
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# 数据划分
# kf = KFold(n_splits=5)
kf = KFold(n_splits=10)
n = kf.get_n_splits(data, target)

for train_index, test_index in kf.split(data, target):
    train_data = data[train_index]
    train_target = target[train_index]
    test_data = data[test_index]
    test_target = target[test_index]

    # 模型训练
    reg = LinearRegression()
    reg = SGDRegressor()
    reg.fit(train_data, train_target)

# 模型评估
#交叉验证
score = cross_val_score(reg,train_data,train_target,cv=kf).mean()
print("交叉验证:{}".format(score))
#平均绝对误差
pre = reg.predict(test_data)
error = mean_squared_error(test_target, pre)
print("平均绝对误差:{}".format(error))
#均方差
error = mean_absolute_error(test_target,pre)
print("均方差:{}".format(error))