# 无监督学习 对没有事先标记过的训练数据进行分类或分群
# 算法不受监督信息(偏见)的约束，则它可能会考虑到一些新的信息
# 不需要事先标记训练数据，极大程度扩大数据样本
# 主要应用：聚类分析、关联规则、维度压缩
# 聚类分析：根据数据某些属性的相似度，将数据分为不同的组（用户画像、商品分类等）
# 关联规则：发现数据中频繁出现的项集和关联规则（购物篮分析、推荐系统等）
# 维度压缩：将高维数据映射到低维空间，保留数据的主要特征（如主成分分析、t-SNE等）

# k-means 均值分类
# 基本思想：将数据分为 k 个簇，每个簇内的数据点相似度较高，不同簇之间的数据点相似度较低
# 算法流程：
# 1. 随机选择 k 个数据点作为初始聚类中心
# 2. 计算每个数据点到聚类中心的距离，将数据点分配到距离最近的聚类中心所在的簇
# 3. 对每个簇内的数据点，计算它们的均值作为新的聚类中心
# 4. 重复步骤 2 和 3，直到聚类中心收敛或达到最大迭代次数
# 特点：实现简单，收敛快；需要指定类别数量

# knn k-近邻分类(k-nearest neighbors, 监督学习)
# 给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例(即k个邻居)
# 这k个实例的多数属于某个类，就把该输入实例分类到这个类中


# meanshift 均值漂移聚类
# 基于密度梯度上升的聚类算法，沿着密度上升方向寻找聚类中心点
# 均值偏移： M(x) = [(u - x0) + (u - x1) + ... + (u - xn)] / n
# 其中，u 是当前数据点，x0, x1, ..., xn 是当前簇内的其他数据点
# 中心更新：u(t+1) = M(t) + u(t)
# 其中，M(t) 是当前数据点的密度梯度，u(t) 是当前数据点的位置, u(t+1) 是更新后的数据点位置
# 1. 随机选择未分类的点作为中心点
# 2. 找出离中心点距离在半径R之内的点，记作集合S
# 3. 计算从中心点到集合S中所有点距离的均值M
# 4. 中心点向M移动，更新其位置
# 5. 重复步骤 2、3、4，直到收敛
# 6. 重复步骤 2-5，直到所有点都被分类
# 7. 分类,根据每个分类,对每个点的访问频率,取访问频率最大的那个类作为当前点集的所属类

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


def show_data(X, y, title, show=False, func=None):
    # plt.figure(figsize=(10, 6))
    l1 = plt.scatter(X.loc[:, "V1"][y == 0], X.loc[:, "V2"][y == 0])
    l2 = plt.scatter(X.loc[:, "V1"][y == 1], X.loc[:, "V2"][y == 1])
    l3 = plt.scatter(X.loc[:, "V1"][y == 2], X.loc[:, "V2"][y == 2])
    plt.title(title)
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.legend([l1, l2, l3], ["cluster 0", "cluster 1", "cluster 2"])
    if func is not None:
        func()
    if show:
        plt.show()


def compare_data(X, y, y_pred, title="origin data", func=None):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    show_data(X, y, "origin data")
    plt.subplot(1, 2, 2)
    show_data(X, y_pred, title, show=True, func=func)


def kmeans(data):
    from sklearn.cluster import KMeans

    X = data.drop(["labels"], axis=1)
    y = data.loc[:, "labels"]
    kmeans = KMeans(n_clusters=3, random_state=6).fit(X)
    centers = kmeans.cluster_centers_
    # 中心点
    print("kmeans centers:", centers)
    y_pred = kmeans.predict(X)
    # 计算准确率
    acc = accuracy_score(y, y_pred)
    print("kmeans accuracy:", acc)
    # 校正分类标签
    y_checked = np.where(y_pred == 0, 1, np.where(y_pred == 1, 2, 0))
    acc = accuracy_score(y, y_checked)
    print("checked kmeans accuracy:", acc)
    compare_data(
        X,
        y,
        y_checked,
        title="k-means clustering",
        func=lambda: plt.scatter(centers[:, 0], centers[:, 1], color="r"),
    )


def knn(data):
    from sklearn.neighbors import KNeighborsClassifier

    X = data.drop(["labels"], axis=1)
    y = data.loc[:, "labels"]
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    y_pred = knn.predict(X)
    print("knn accuracy:", accuracy_score(y, y_pred))
    compare_data(X, y, y_pred, title="knn clustering")


def meanshift(data):
    from sklearn.cluster import MeanShift
    from sklearn.cluster import estimate_bandwidth

    X = data.drop(["labels"], axis=1)
    y = data.loc[:, "labels"]

    # 计算带宽
    bandwidth = estimate_bandwidth(X, n_samples=300)
    print("bandwidth:", bandwidth)

    meanshift = MeanShift(bandwidth=bandwidth).fit(X)
    y_pred = meanshift.predict(X)
    print("meanshift accuracy:", accuracy_score(y, y_pred))
    y_checked = np.where(y_pred == 0, 2, np.where(y_pred == 2, 0, 1))
    print("checked meanshift accuracy:", accuracy_score(y, y_checked))
    compare_data(X, y, y_checked, title="meanshift clustering")


if __name__ == "__main__":
    data = pd.read_csv("data/kmeans_knn_meanshift_data.csv")
    meanshift(data)
