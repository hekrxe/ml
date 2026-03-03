import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽TensorFlow警告

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense, Input
from keras.callbacks import Callback
from keras.optimizers import Adam


# 建立MLP,在不增加特征项的前提下实现非线性二分类


def show_data(X, y, x_range, y_range_form):
    plt.figure(figsize=(6, 6))
    # 先画决策边界（背景）
    plt.scatter(
        x_range[:, 0][y_range_form == 1],
        x_range[:, 1][y_range_form == 1],
        c="limegreen",
        s=1,
        alpha=0.6,
        label="MLP Y",
    )
    plt.scatter(
        x_range[:, 0][y_range_form == 0],
        x_range[:, 1][y_range_form == 0],
        c="indianred",
        s=1,
        alpha=0.6,
        label="MLP N",
    )

    # 再画原始数据点（在上面）
    plt.scatter(
        X.loc[:, "x1"][y == 1],
        X.loc[:, "x2"][y == 1],
        c="green",
        s=50,
        edgecolors="black",
        linewidth=1,
        label="Y",
    )
    plt.scatter(
        X.loc[:, "x1"][y == 0],
        X.loc[:, "x2"][y == 0],
        c="red",
        s=50,
        edgecolors="black",
        linewidth=1,
        label="N",
    )

    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("MLP Classification")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/data-mlp-01.csv")
    X = data.drop(["y"], axis=1)
    y = data.loc[:, "y"]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=10
    )
    mlp = Sequential()
    # 通过add叠加各网络层
    mlp.add(Input(shape=(2,)))  # 输入层，2个特征（x1, x2）
    # 隐藏层, 10个神经元, 激活函数, 二分类常用sigmoid函数
    mlp.add(Dense(units=10, activation="sigmoid"))
    # 输出层
    mlp.add(Dense(units=1, activation="sigmoid"))
    # 查看模型结构
    mlp.summary()
    # 二分类交叉熵损失函数, adam优化器
    mlp.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.01))

    class PrintLast10Logs(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: {logs}")

    from keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor="loss", patience=50, restore_best_weights=True  # 20轮没有改善就停止
    )

    mlp.fit(
        X_train,
        y_train,
        epochs=3000,
        verbose=0,
        callbacks=[PrintLast10Logs(), early_stop],
        batch_size=32,
    )

    y_train_pred = (mlp.predict(X_train) > 0.5).astype("int32")
    print("train acc: ", accuracy_score(y_train, y_train_pred))
    y_test_pred = (mlp.predict(X_test) > 0.5).astype("int32")
    print("test acc: ", accuracy_score(y_test, y_test_pred))

    xx, yy = np.meshgrid(np.arange(0, 1, 0.03), np.arange(0, 1, 0.03))
    x_range = np.c_[xx.ravel(), yy.ravel()]  # 点积
    y_range_pred = (mlp.predict(x_range) > 0.5).astype("int32")

    # format y_range_pred
    y_range_pred_form = pd.Series(i[0] for i in y_range_pred)

    show_data(X, y, x_range, y_range_pred_form)
