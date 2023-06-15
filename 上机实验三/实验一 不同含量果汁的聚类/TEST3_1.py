# -*- coding = utf-8 -*-
# UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
import os

os.environ["OMP_NUM_THREADS"] = '1'

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# 导入数据
data = pd.read_csv('./dataset/4_beverage.csv')

# 画散点图，k可能为4
plt.figure()
plt.grid()
plt.scatter(data['juice'], data['sweet'])
plt.xlabel('juice')
plt.ylabel('sweet')
plt.title('beverage')
plt.show()

# 模型训练
SC = []
for K in range(2, 11):
    kmeans = KMeans(n_clusters=K, n_init='auto')
    kmeans.fit(data)
    pre = kmeans.predict(data)

    # 聚类
    plt.figure()
    plt.scatter(data['juice'], data['sweet'], c=pre)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r', marker='x')
    plt.xlabel('juice')
    plt.ylabel('sweet')
    plt.title('K={}'.format(K))
    plt.show()

    # 模型评估
    pre = kmeans.predict(data)
    score = calinski_harabasz_score(data, pre)
    SC.append(score)
    print('K={}时,Calinski-Harabasz Score: '.format(K), score)

# 不同K值的Calinski-Harabasz Score
X = [i for i in range(2, 11)]
plt.figure()
plt.grid()
plt.plot(X, SC)
plt.xlabel('K')
plt.ylabel('Calinski-Harabasz Score')
plt.show()
