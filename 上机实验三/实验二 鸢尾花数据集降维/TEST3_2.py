# -*- coding = utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 导入数据集
dataset = load_iris()
data = dataset['data']
target = dataset['target']
target_names = dataset['target_names']

# 模型训练
for K in range(1, 5):
    pca = PCA(n_components=K)
    pca.fit(data)
    new_data = pca.transform(data)

    # 可视化
    if K == 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(new_data[:, 0], new_data[:, 1], c=target, s=50)
        handles, labels = scatter.legend_elements()
        labels = '$\\mathdefault{setosa}$', '$\\mathdefault{versicolor}$', '$\\mathdefault{virginica}$'
        legend2 = ax.legend(handles, labels, loc='best', title='types')
        plt.show()

    print('K={}时：'.format(K))
    print('查看降维后每个新特征向量上所带的信息量大小:', pca.explained_variance_)
    print('每个新特征向量所占的信息量占原始数据总信息量的百分比:', pca.explained_variance_ratio_)


    if K == 4:
        plt.figure()
        plt.plot([1, 2, 3, 4], np.cumsum(pca.explained_variance_ratio_))
        plt.xticks([1, 2, 3, 4])
        plt.xlabel('number of components after dimension reduction')
        plt.ylabel('cumulative explained variance ratio')
        plt.show()

    train_data, test_data, train_target, test_target = train_test_split(new_data, target, train_size=0.7)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(new_data, target)
    pre = clf.predict(test_data)
    print('n={}时，准确率为：'.format(K), accuracy_score(test_target, pre))
