# PCA
# 主成分分析(Principal Component Analysis)是一种常用的降维技术。
# 目标：寻找k(k<n)维新数据，使他们反映事物的主要特征
# 核心：在信息损失尽可能少的情况下，降低数据维度
# 如何保留主要信息：投影后的不同特征数据尽可能分得开(即不相关)
# 如何实现：使投影后数据的方差最大,因为方差越大数据也越分散
# 步骤：
# 1. 对数据进行标准化处理，使每个特征的**均值为0，方差为1**
# 2. 计算协方差矩阵特征向量、及数据在各特征向量投影后的方差
# 3. 根据需求(任务指定或方差比例)确定降维维度k
# 4. 选择k维特征向量,计算数据在其形成空间的投影

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def show_origin(X, X_norm):
    x_mean = X.loc[:, "sepal length"].mean()
    x_sigma = X.loc[:, "sepal length"].std()
    x_norm_mean = X_norm[:, 0].mean()
    x_norm_sigma = X_norm[:, 0].std()

    print("x_mean:", x_mean)
    print("x_sigma:", x_sigma)
    print("x_norm_mean:", x_norm_mean)
    print("x_norm_sigma:", x_norm_sigma)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(X.loc[:, "sepal length"], bins=100)
    plt.title("sepal length")

    plt.subplot(1, 2, 2)
    plt.hist(X_norm[:, 0], bins=100)
    plt.title("sepal length normalized")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/iris_data.csv")
    X = data.drop(["target", "label"], axis=1)
    y = data.loc[:, "label"]
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    y_pred = knn.predict(X)
    print("knn accuracy:", accuracy_score(y, y_pred))
    # 数据标准化
    X_norm = StandardScaler().fit_transform(X)
    # show_origin(X, X_norm)

    # pca = PCA(n_components=4)
    # X_pca = pca.fit_transform(X_norm)
    # 查看降维后数据的方差占比
    # var_ratio = pca.explained_variance_ratio_
    # print("var_ratio:", var_ratio)
    # plt.figure(figsize=(6, 4))
    # plt.bar([1, 2, 3, 4], var_ratio)
    # plt.title("variance ratio of each principal component")
    # plt.xticks([1, 2, 3, 4], ["PC1", "PC2", "PC3", "PC4"])
    # plt.ylabel("variance ratio")
    # plt.show()

    # 保留合适的主成分，可视化降维后的数据
    # 选择保留前两个主成分
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)  # 降维
    print("X_pca:", X_pca.shape)

    plt.figure(figsize=(8, 6))
    setosa = plt.scatter(
        X_pca[:, 0][y == 0], X_pca[:, 1][y == 0], c="r", label="setosa"
    )
    versicolor = plt.scatter(
        X_pca[:, 0][y == 1], X_pca[:, 1][y == 1], c="g", label="versicolor"
    )
    virginica = plt.scatter(
        X_pca[:, 0][y == 2], X_pca[:, 1][y == 2], c="b", label="virginica"
    )
    plt.title("2D projection of iris data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(handles=[setosa, versicolor, virginica])
    plt.show()

    # 基于降维后的特征进行knn分类
    knn_pca = KNeighborsClassifier(n_neighbors=3).fit(X_pca, y)
    y_pred = knn_pca.predict(X_pca)
    print("knn_pca accuracy:", accuracy_score(y, y_pred))
