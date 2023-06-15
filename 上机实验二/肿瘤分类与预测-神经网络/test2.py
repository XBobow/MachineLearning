from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
# 导入数据集
dataset = datasets.load_breast_cancer()
data = dataset['data']
target = dataset['target']

# 数据划分
train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=(0.8))

# 模型训练
# 分析不同参数对分类结果的影响，包括：
# 1.网络结构 hidden_layer_sizes
# 2.优化器solver
# 3.激活函数 activation
# 4.最大迭代次数 max_iter
acc = []
recall = []
f1 = []
start = time.time()
for i in range(100):
    mlp = MLPClassifier(hidden_layer_sizes=(14,32,8),solver='adam',activation='relu',max_iter=1000)
    mlp.fit(train_data, train_target)

    # 模型评估
    pre = mlp.predict(test_data)
    print(pre)
    acc.append(accuracy_score(test_target, pre))
    recall.append(recall_score(test_target, pre, average='macro'))
    f1.append(f1_score(test_target, pre, average='macro'))
    print(i)
    print('accuracy_score:', accuracy_score(test_target, pre))
    print('recall_score:', recall_score(test_target, pre, average='macro'))
    print('f1_score:', f1_score(test_target, pre, average='macro'))

end = time.time()


print('\n\n\n')
print('each_time:{}s'.format((end-start)/100))
print('accuracy_score_mean:', np.mean(acc))
print('recall_score_mean:', np.mean(recall))
print('f1_score_mean:', np.mean(f1))
print('accuracy_score_min:', np.min(acc))
print('recall_score_min:', np.min(recall))
print('f1_score_min:', np.min(f1))
print('accuracy_score_max:', np.max(acc))
print('recall_score_max:', np.max(recall))
print('f1_score_max:', np.max(f1))
print('accuracy_score_std:', np.std(acc))
print('recall_score_std:', np.std(recall))
print('f1_score_std:', np.std(f1))

# 模型预测
# 训练数据
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
# fig.add_axes(ax)
# ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], marker='o', c=train_target)
# plt.show()
# # 测试数据
# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
# fig.add_axes(ax)
# ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], marker='o', c=pre)
# plt.show()
