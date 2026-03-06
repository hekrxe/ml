# 迁移学习
# 1. 特征提取
#   1.1 使用模型A,移除输出层,提取目标特征信息
# 2. 结构引用
#   2.1 使用模型A的结构,重新/二次训练权重系数参数
# 3. 部分训练
#   3.1 使用模型A的结构,重新训练部分层的权重系数参数
#
#   ^ 数据量
#   |               |
#   |     全新模型   |  结构引用
#   |               |
#   |   -------------------------
#   |               |
#   |     部分训练   |  特征提取
#   |               |
#   |------------------------------>
#                           任务相似度

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


def show_pred(X, y, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y)
    plt.plot(X, y_pred, color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("transfer data")
    plt.show()


def model1_fix():

    data = pd.read_csv("data/transfer_data.csv")
    X = data.loc[:, "x"]
    y = data.loc[:, "y"]

    def show(x, y):
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("transfer data")
        plt.show()

    # show(X, y)

    X = np.array(X).reshape(-1, 1)

    from keras.models import Sequential
    from keras.layers import Dense, Input

    model1 = Sequential()
    model1.add(Input(shape=(1,)))
    model1.add(Dense(units=50, activation="relu"))
    model1.add(Dense(units=50, activation="relu"))
    model1.add(Dense(units=1, activation="linear"))
    model1.compile(optimizer="adam", loss="mean_squared_error")
    model1.summary()

    model1.fit(X, y, epochs=50, batch_size=2)

    y_pred = model1.predict(X)

    show_pred(X, y, y_pred)
    joblib.dump(model1, "model/transfer_model_1.m")


def load_model1_and_fit():
    model2 = joblib.load("model/transfer_model_1.m")
    data2 = pd.read_csv("data/transfer_data2.csv")
    X2 = data2.loc[:, "x2"]
    y2 = data2.loc[:, "y2"]

    X2 = np.array(X2).reshape(-1, 1)
    # transfer learning
    model2.fit(X2, y2, epochs=50, batch_size=2)
    y2_pred = model2.predict(X2)
    show_pred(X2, y2, y2_pred)


if __name__ == "__main__":
    # model1_fix()
    load_model1_and_fit()
