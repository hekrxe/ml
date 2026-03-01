# 决策树
# 一种对实例进行分类的树形结构，通过多层判断区分目标所属类别
# 本质：通过多层判断，从训练数据集中归纳出一组分类规则
# 优点：计算量小，运算速度快；易于理解，可清晰查看各属性的重要性
# 缺点：忽略属性间的相关性；样本类别分布不均匀时，容易影响模型表现
# 目标：根据训练数据集构建一个决策树模型
# 问题核心：特征选择，每一个节点，应该选用哪个特征
# 三种求解方法：ID3、C4.5、CART
# ID3：
# 利用信息熵原理选择信息增益最大的属性作为分类属性，递归地拓展决策树的分枝，完成决策树的构造
#   信息熵：
#   表示随机变量不确定性的度量，熵越大，随机变量的不确定性就越大
#   信息增益：
#   表示使用一个属性对样本进行分类时，能够带来的信息增益
#   信息增益 = 熵 - 条件熵
#   条件熵：
#   表示在已知某个属性的情况下，随机变量的不确定性
#   信息增益最大的属性，就是能够最好地区分样本类别的属性
#   目标：划分后样本分布不确定性尽可能小，即划分后信息熵小，信息增益大

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("data/iris_data.csv")
    X = data.drop(["target", "label"], axis=1)
    y = data.loc[:, "target"]
    tclf = tree.DecisionTreeClassifier(
        criterion="entropy",  # 信息增益最大化
        min_samples_leaf=5,  #
    ).fit(X, y)
    y_pred = tclf.predict(X)
    acc = accuracy_score(y, y_pred)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print(tclf)

    plt.figure(figsize=(8, 8))
    tree.plot_tree(
        tclf,
        filled=True,
        feature_names=X.columns,
        class_names=y.unique(),
    )
    plt.show()
