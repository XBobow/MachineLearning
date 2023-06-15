# -*- coding = utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# missing from current font
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 导入数据集
dataset = load_diabetes()
data = dataset['data']
target = dataset['target']

# 数据标准化
data = StandardScaler().fit_transform(data)

# 数据划分
train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=(0.7))

# 模型训练
MSE = []
for i in np.arange(0.01, 100, 0.01):
    ridge = Ridge(alpha=i)
    ridge.fit(train_data, train_target)
    ridge_pre = ridge.predict(test_data)
    MSE.append(mean_squared_error(test_target, ridge_pre))
    # print('ridge mean_squared_error:', mean_squared_error(test_target, ridge_pre))

plt.figure()
plt.plot(np.arange(0.01, 100, 0.01), MSE)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title('岭回归不同alpha参数模型性能')
plt.show()

MSE.clear()
FEATURE_NUMBERS = []
for i in np.arange(0.01, 2, 0.01):
    lasso = Lasso(alpha=i)
    lasso.fit(train_data, train_target)
    lasso_pre = lasso.predict(test_data)
    MSE.append(mean_squared_error(test_target, lasso_pre))
    FEATURE_NUMBERS.append(np.sum(lasso.coef_ != 0))
    # print('lasso mean_squared_error:', mean_squared_error(test_target, lasso_pre))

plt.figure()
plt.plot(np.arange(0.01, 2, 0.01), MSE)
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.title('Lasso回归不同alpha参数模型性能')
plt.show()

plt.figure()
plt.plot(np.arange(0.01, 2, 0.01), FEATURE_NUMBERS)
plt.xlabel('alpha')
plt.ylabel('FEATURE_NUMBERS')
plt.title('Lasso回归不同alpha参数模型特征数')
plt.show()
