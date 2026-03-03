# 酶活性预测
# 判断模型优劣

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def show_lr_data(X_train, y_train, X_test, y_test, X_range, y_range_pred):
    plt.figure()
    plt.plot(X_range, y_range_pred)
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.xlabel("Temperature")
    plt.ylabel("Rate")
    plt.title("T-R")
    plt.show()


def linear_model_fit(X_train, y_train, X_test, y_test, X_range):
    l = LinearRegression().fit(X_train, y_train)
    # 进行预测
    y_train_pred = l.predict(X_train)
    y_test_pred = l.predict(X_test)
    r2_train_score = r2_score(y_train, y_train_pred)
    r2_test_score = r2_score(y_test, y_test_pred)
    print("r2 train score: ", r2_train_score, ", r2 test score: ", r2_test_score)
    return l.predict(X_range)


# 线性回归 过拟合 欠拟合
def lr_temprature():
    data = pd.read_csv("data/T-R-train.csv")
    X_train = data.loc[:, "T"]
    y_train = data.loc[:, "rate"]
    # 转换为n行1列, 因为sklearn要求输入为二维数组, 而X_train为一维数组
    X_train = np.array(X_train).reshape(-1, 1)

    data_test = pd.read_csv("data/T-R-test.csv")
    X_test = data_test.loc[:, "T"]
    y_test = data_test.loc[:, "rate"]
    X_test = np.array(X_test).reshape(-1, 1)

    X_range = np.linspace(40, 90, 300).reshape(-1, 1)

    # 欠拟合
    y_range_pred = linear_model_fit(X_train, y_train, X_test, y_test, X_range)
    show_lr_data(X_train, y_train, X_test, y_test, X_range, y_range_pred)

    # 拟合
    def fitting(x):
        # 转为多项式模型
        # 2阶
        poly = PolynomialFeatures(degree=x)
        # 即 x -> ax^2 + bx + c
        X_x_train = poly.fit_transform(X_train)
        X_x_test = poly.transform(X_test)
        X_x_range = poly.transform(X_range)
        y_x_range_pred = linear_model_fit(
            X_x_train, y_train, X_x_test, y_test, X_x_range
        )
        show_lr_data(X_train, y_train, X_test, y_test, X_range, y_x_range_pred)

    fitting(2)  # 正常拟合
    fitting(5)  # 过拟合


# 基于高斯密度函数,寻找异常点并剔除
def outlier_filtering_by_gaussian_density():
    data = pd.read_csv("data/data_class_raw.csv")
    X = data.drop(["y"], axis=1)
    y = data.loc[:, "y"]

    # anomaly detection
    ad = EllipticEnvelope(contamination=0.02)
    ad.fit(X[y == 0])
    y_bad_pred = ad.predict(X[y == 0])
    print(y_bad_pred)

    plt.figure()
    bad = plt.scatter(X.loc[:, "x1"][y == 0], X.loc[:, "x2"][y == 0])
    good = plt.scatter(X.loc[:, "x1"][y == 1], X.loc[:, "x2"][y == 1])
    # 找出异常点
    plt.scatter(
        X.loc[:, "x1"][y == 0][y_bad_pred == -1],
        X.loc[:, "x2"][y == 0][y_bad_pred == -1],
        marker="x",
        c="r",
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("gaussian_density")
    plt.legend((bad, good), ("bad", "good"))
    plt.show()
    # 根据异常点, 剔除异常点后进行PCA处理


# 基于 outlier_filtering_by_gaussian_density 剔除异常点后进行PCA处理
# 处理后的数据是 data/data_class_processed.csv
def pca_knn():
    # 读取处理后的数据
    data = pd.read_csv("data/data_class_processed.csv")
    X = data.drop(["y"], axis=1)
    y = data.loc[:, "y"]
    # 异常点剔除后, 进行PCA处理
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    # 进行PCA降维
    X_reduced = pca.fit_transform(X_norm)
    var_ratio = pca.explained_variance_ratio_
    print("explained variance ratio: ", var_ratio, X_reduced)
    # explained variance ratio:  [0.5369408 0.4630592]
    # 第一个主成分解释了53.7%的标准差, 第二个主成分解释了46.3%的标准差
    # 可以看到两个维度上的主成分标准差都很高, 因此都需要保留 不能降维

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,  # 这里即可以使用原始数据也可以使用降维后的数据, 因为两个维度上的主成分标准差都很高
        y,
        test_size=0.4,
        random_state=4,
    )
    print("shape of X_train: ", X_train.shape)
    print("shape of X_test: ", X_test.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of y_test: ", y_test.shape)

    # 200*200
    xx, yy = np.meshgrid(np.arange(0, 10, 0.05), np.arange(0, 10, 0.05))
    # 转换为若干行2列
    # 40000*2
    X_range = np.c_[xx.ravel(), yy.ravel()]

    def knn_x_fitting(x):
        print("-" * 25)
        print("k: ", x)
        # 建立knn模型进行分类
        knn_10 = KNeighborsClassifier(n_neighbors=x)
        knn_10.fit(X_train, y_train)
        y_train_pred = knn_10.predict(X_train)
        y_test_pred = knn_10.predict(X_test)
        # 计算准确率
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("train accuracy: ", train_accuracy, ", test accuracy: ", test_accuracy)

        # 进行预测
        y_range_pred = knn_10.predict(X_range)

        # 可视化
        plt.figure()
        kbad = plt.scatter(
            X_range[:, 0][y_range_pred == 0], X_range[:, 1][y_range_pred == 0]
        )
        kgood = plt.scatter(
            X_range[:, 0][y_range_pred == 1], X_range[:, 1][y_range_pred == 1]
        )

        bad = plt.scatter(X.loc[:, "x1"][y == 0], X.loc[:, "x2"][y == 0])
        good = plt.scatter(X.loc[:, "x1"][y == 1], X.loc[:, "x2"][y == 1])
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("knn_" + str(x))
        plt.legend((kbad, kgood, bad, good), ("kbad", "kgood", "bad", "good"))
        plt.show()

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_test_pred)
        print("confusion matrix: ", cm)
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Recall = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        Precision = TP / (TP + FP)
        F1_score = 2 * TP / (2 * TP + FP + FN)

        print("Accuracy: ", Accuracy)
        print("Recall: ", Recall)
        print("Specificity: ", Specificity)
        print("Precision: ", Precision)
        # 2 * Precision * Recall / (Precision + Recall)
        print("F1 score: ", F1_score)

        return test_accuracy

    # 记录每个k值对应的测试集准确率
    test_accuracies = []
    ks = [1, 5, 10, 15, 20]

    # 进行k=10, 20, 30, 40, 50的knn分类
    for x in ks:
        test_accuracy = knn_x_fitting(x)
        test_accuracies.append(test_accuracy)

    # 可视化每个k值对应的测试集准确率
    plt.figure()
    plt.plot(ks, test_accuracies, marker="o")
    plt.xlabel("k")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. k")
    plt.show()


if __name__ == "__main__":
    pca_knn()
