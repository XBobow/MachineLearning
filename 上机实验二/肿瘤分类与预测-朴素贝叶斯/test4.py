from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 导入数据集
dataset = datasets.load_breast_cancer()
data = dataset['data']
target = dataset['target']

# 画散点图
plt.figure()
for i in range(30):
    plt.grid()
    plt.scatter(data[:, i], target)
    plt.title(dataset['feature_names'][i])
    plt.show()
    plt.clf()
plt.close()

accuracy_score_list = []
recall_score_list = []
f1_score_list = []
# 数据划分
# 尝试不同的数据集划分比例（如9:1,8:2,7:3,…,1:9），分析对预测结果的影响，画出性能关于不同比例的折线图
for i in range(1, 10):
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=(1 - i / 10))

    # 数据归一化
    # 不需要归一化

    # 模型训练
    GNB = GaussianNB()
    GNB.fit(train_data, train_target)

    # 模型评估
    pre = GNB.predict(test_data)
    print(pre)
    accuracyscore = accuracy_score(test_target, pre)
    recallscore = recall_score(test_target, pre, average='macro')
    f1score = f1_score(test_target, pre, average='macro')
    accuracy_score_list.append(accuracyscore)
    recall_score_list.append(recallscore)
    f1_score_list.append(f1score)
    print('accuracy_score:', accuracyscore)
    print('recall_score:', recallscore)
    print('f1_score:', f1score)
    print('confusion_matrix:', confusion_matrix(test_target, pre))
    confusionmatrix = confusion_matrix(test_target, pre)
    plt.figure()
    sns.heatmap(confusionmatrix, annot=True, fmt='.2f')

# 性能关于不同比例的折线图
x = [str(10 - i) + ':' + str(i) for i in range(1, 10)]
y = [accuracy_score_list, recall_score_list, f1_score_list]
titles = ['accuracy_score', 'recall_score', 'f1_score']
for i in range(3):
    plt.figure()
    plt.grid()
    plt.plot(x, y[i])
    plt.title(titles[i])
plt.show()
