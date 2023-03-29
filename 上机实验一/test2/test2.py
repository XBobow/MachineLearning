from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# 导入数据集
dataset = datasets.load_iris()

data = dataset['data']
target = dataset['target']
feature = dataset['feature_names']



#画散点图
for i in range(4):
    plt.figure()
    plt.grid()
    plt.scatter(data[:,i],target)
    plt.title(feature[i])
plt.show()

#选择鸢尾花类别
data = data[:100, ]
target = target[:100, ]


acc = []
for i in range(1, 10):
    # 数据划分
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=(i * 0.1))

    # 数据归一化
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    # 模型训练
    re = LogisticRegression()
    re.fit(train_data, train_target)

    # 模型评估
    pre = re.predict(test_data)
    acc.append(accuracy_score(test_target, pre))

# 绘制折线图
x= []
for i in range(1,10):
    x.append(i*0.1)
plt.figure()
plt.plot(x,acc)
plt.title('accuracy_score')
plt.show()
