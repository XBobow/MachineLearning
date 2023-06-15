from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

# 导入数据集
dataset = datasets.load_wine()
data = dataset['data']
target = dataset['target']

# 画散点图
# for i in range(13):
#     plt.figure()
#     plt.grid()
#     plt.scatter(data[:, i], target)
#     plt.title(dataset['feature_names'][i])
#     plt.savefig('pics/wine_' + str(i) + '.png')
# plt.show()

# 数据划分
train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=(0.7))

# 数据归一化
# 不需要归一化

# 模型训练
# 分析不同参数对分类结果的影响，包括：
# 1.不同的决策树不纯度计算方法criterison，信息熵entropy，基尼指数gini。
# 2.控制决策树中的随机选项splitter,取值best、random。
# 3.限制决策树的最大深度，超过设定深度的树枝全部剪掉max_depth。
# 4.节点在分之后每个子节点至少包含的训练样本数min_samples_leaf
# 5.节点允许被分支至少需要包含的训练样本数min_samples_split
dt = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=3, min_samples_split=5,min_samples_leaf=3)

# dt = DecisionTreeClassifier(criterion='gini')
import numpy as np
acc = []
recall = []
import time

acc_mean = []
acc_min = []
acc_max = []
acc_std = []
recall_mean = []
recall_min = []
recall_max = []
recall_std = []

start = time.time()
for X in range(100):
    acc.clear()
    recall.clear()
    for i in range(1000):
        dt = DecisionTreeClassifier(criterion='entropy', splitter='best',max_depth=3,min_samples_leaf=6,min_samples_split=X+1)
        dt.fit(train_data, train_target)

        # 模型评估
        pre = dt.predict(test_data)
        print(i)
        print('accuracy_score:', accuracy_score(test_target, pre))
        print('recall_score:', recall_score(test_target, pre, average='macro'))
        acc.append(accuracy_score(test_target, pre))
        recall.append(recall_score(test_target, pre, average='macro'))


    acc_mean.append(np.mean(acc))
    acc_min.append(np.min(acc))
    acc_max.append(np.max(acc))
    acc_std.append(np.std(acc))
    recall_mean.append(np.mean(recall))
    recall_min.append(np.min(recall))
    recall_max.append(np.max(recall))
    recall_std.append(np.std(recall))

X = [i+1 for i in range(100)]
Y = [acc_mean, acc_min, acc_max, acc_std, recall_mean, recall_min, recall_max, recall_std]
T = ['acc_mean', 'acc_min', 'acc_max', 'acc_std', 'recall_mean', 'recall_min', 'recall_max', 'recall_std']
for i in range(8):
    plt.figure()
    plt.grid()
    plt.plot(X, Y[i])
    plt.title(T[i])
    plt.savefig('pics/wine_' + str(i*10) + '.png')
plt.show()
    # print('\n\n\n')
    # print('accuracy_score_mean:', np.mean(acc))
    # print('recall_score_mean:', np.mean(recall))
    # print('accuracy_score_min:', np.min(acc))
    # print('recall_score_min:', np.min(recall))
    # print('accuracy_score_max:', np.max(acc))
    # print('recall_score_max:', np.max(recall))
    # print('accuracy_score_std:', np.std(acc))
    # print('recall_score_std:', np.std(recall))

