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
# feature = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
# for i in range(13):
#     plt.figure(figsize=(10, 7))
#     plt.grid()
#     plt.scatter(data[:, i], target, s=5)
#     plt.title(feature[i])
#     # print(feature[i],np.corrcoef(data[:i]),target)
# plt.show()

# 数据归一化
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

# 数据划分
kf = KFold(n_splits=5)
kf = KFold(n_splits=10)
n = kf.get_n_splits(data, target)
# for i, (train_index, test_index) in enumerate(kf.split(data,target)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")
for train_index, test_index in kf.split(data, target):
    train_data = data[train_index]
    train_target = target[train_index]
    test_data = data[test_index]
    test_target = target[test_index]

# 模型训练
reg = LinearRegression()
reg.fit(train_data, train_target)
# reg.score(train_data, train_target)
# reg.coef_
# print(reg.predict(test_data))

# 模型评估
score = cross_val_score(reg,train_data,train_target,cv=kf,scoring="neg_mean_squared_error").mean()
score = cross_val_score(reg,train_data,train_target,cv=kf,scoring="neg_mean_absolute_error").mean()
print(-score)

# pre = reg.predict(test_data)
# error = mean_squared_error(test_target, pre)
# error = mean_absolute_error(test_target,pre)
# print(error)
