# 异常检测
# 根据输入数据，对不符合预期模式的数据进行识别。
# 概率密度
# 概率密度函数是一个描述随机变量在某个确定的取值点附近的可能性的函数。
# 高斯分布（也称为正态分布）是一种连续概率分布，用于描述随机变量在一个区间内的取值概率。
# 它的概率密度函数是一个钟形曲线，中心对称，且高度最大的点对应于随机变量的均值。

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope


def show_data(x1, x2):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(x1, x2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("anomaly data")

    plt.subplot(1, 3, 2)
    plt.hist(x1, bins=100)
    plt.xlabel("x1")
    plt.title("x1 histogram")

    plt.subplot(1, 3, 3)
    plt.hist(x2, bins=100)
    plt.xlabel("x2")
    plt.title("x2 histogram")
    plt.show()


def show_norm(x1_range, x1_norm, x2_range, x2_norm):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x1_range, x1_norm)
    plt.xlabel("x1")
    plt.ylabel("p(x)")
    plt.title("x1 normal distribution")

    plt.subplot(1, 2, 2)
    plt.plot(x2_range, x2_norm)
    plt.xlabel("x2")
    plt.ylabel("p(x)")
    plt.title("x2 normal distribution")
    plt.show()


def show_result(x1, x2, y_pred):
    plt.figure(figsize=(9, 5))
    plt.scatter(x1, x2, marker="x")
    plt.scatter(
        x1[y_pred == -1],
        x2[y_pred == -1],
        marker="o",
        facecolors="none",
        edgecolors="r",
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("anomaly detection result")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/anomaly_data.csv")
    x1 = data.loc[:, "x1"]
    x2 = data.loc[:, "x2"]
    # show_data(x1, x2)

    # 均值
    x1_mean = x1.mean()
    x2_mean = x2.mean()
    # 标准差
    x1_std = x1.std()
    x2_std = x2.std()
    print(x1_mean, x1_std, x2_mean, x2_std)

    # 计算高斯分布
    x1_range = np.linspace(0, 20, 300)
    # 概率分布函数
    x1_norm = norm.pdf(x1_range, x1_mean, x1_std)
    x2_range = np.linspace(0, 20, 300)
    x2_norm = norm.pdf(x2_range, x2_mean, x2_std)
    # show_norm(x1_range, x1_norm, x2_range, x2_norm)

    model = EllipticEnvelope(contamination=0.019)  # contamination 异常比例
    model.fit(data)
    y_pred = model.predict(data)
    show_result(x1, x2, y_pred)
