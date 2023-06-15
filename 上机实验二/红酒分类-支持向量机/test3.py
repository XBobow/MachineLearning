import time
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score

# 导入数据集
dataset = datasets.load_wine()
data = dataset['data']
target = dataset['target']

# 数据统计
df = {}
for i in range(13):
    df[dataset['feature_names'][i]] = pd.Series(data[:, i])
df = pd.DataFrame(df)
print('样本在特征值下的平均值:\n' + str(df.mean()) + '\n')
print('样本在特征值下的方差:\n' + str(df.var()) + '\n')
print('样本在特征值下的最小值:\n' + str(df.min()) + '\n')
print('样本在特征值下的最大值:\n' + str(df.max()) + '\n')

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)

for i in range(13):
    print(dataset['feature_names'][i] + '的类别情况:')
    print('共有' + str(df[dataset['feature_names'][i]].nunique()) + '种类别')
    d = df[dataset['feature_names'][i]].value_counts()
    d.to_csv('{}.csv'.format(dataset['feature_names'][i]).replace('/', '%'))
    print(d, end='\n\n')

# 绘制热力图
corr = df.corr()
# print('特征值之间的相关性:\n'+str(corr)+'\n')
plt.figure()
seaborn.heatmap(corr, annot=True, vmax=1, square=True)
plt.show()

# 数据划分
train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=(0.8))

# 模型训练
# 分析不同参数对分类结果的影响，包括：
# 1.惩罚参数C
# 2.核函数类型kernel
svc = SVC(kernel='linear', C=1)
start = time.time()
svc.fit(train_data, train_target)
end = time.time()

# 模型评估
pre = svc.predict(test_data)
print(pre)
print('accuracy_score:', accuracy_score(test_target, pre))
print('recall_score:', recall_score(test_target, pre, average='macro'))
print('time:{}s'.format(end - start))

