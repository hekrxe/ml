# 循环神经网络
# 前部序列的信息经过处理后, 作为输入信息传递到后部序列


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


def get_Xy(price, time_step=10):
    X = []
    y = []
    for i in range(len(price) - time_step):
        X.append([e for e in price[i : i + time_step]])
        y.append(price[i + time_step])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, np.array(y)


def show_lr_data(
    y_train_actual,
    y_train_pred,
    y_test_actual,
    y_test_pred,
    last_data,
    future_x_data,
    time_step,
):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(y_train_actual, label="train actual")
    plt.plot(y_train_pred, label="train predict")
    plt.ylabel("nav")
    plt.legend()
    plt.title("fund 017057 train predict")

    plt.subplot(2, 2, 2)
    plt.plot(y_test_actual, label="test actual")
    plt.plot(y_test_pred, label="test predict")
    plt.ylabel("nav")
    plt.legend()
    plt.title("fund 017057 test predict")

    plt.subplot(2, 2, 4)
    # 将历史数据和预测数据连接起来
    len_combined = len(last_data) + len(future_x_data)
    # 绘制历史数据（蓝色，实线）
    plt.scatter(
        range(len(last_data)),
        last_data,
        label="last {} data".format(time_step),
        color="blue",
    )
    # 绘制预测数据（红色，虚线），从最后一个历史数据点开始
    plt.scatter(
        range(len(last_data), len_combined),
        future_x_data,
        label="future x data",
        color="red",
        linestyle="--",
    )
    plt.ylabel("nav")
    plt.legend()
    plt.title("fund 017057 future x data")
    plt.tight_layout()
    plt.show()


def predict_future(model, last_data, max_price, days=5):
    predictions = []
    current_input = last_data.copy()
    for _ in range(days):
        x_input = current_input.reshape(1, len(current_input), 1)
        pred = model.predict(x_input)[0][0]
        print("last:", current_input * max_price, "pred:", pred * max_price)
        predictions.append(pred)
        current_input = np.append(current_input[1:], pred)
    return np.array(predictions)


if __name__ == "__main__":
    data = pd.read_csv("data/fund_017057.csv")
    price = data.loc[:, "unit_nav"]
    max_price = max(price)
    price = price / max_price
    time_step = 9
    X, y = get_Xy(price, time_step)

    # 分割数据
    split_index = int(len(X) * 0.85)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(SimpleRNN(units=11, activation="relu"))
    model.add(Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    model.fit(X_train, y_train, epochs=301, batch_size=16)

    y_train_pred = model.predict(X_train) * max_price
    y_train_actual = y_train * max_price

    y_test_pred = model.predict(X_test) * max_price
    y_test_actual = y_test * max_price

    # 预测未来x天 & 反归一化
    future_x_data = predict_future(model, y[-time_step:], max_price, days=5)
    print("y_test_actual:", y_test_actual[-time_step:])
    print("------------y:", y[-time_step:] * max_price)
    print("future x data:", future_x_data * max_price)
    show_lr_data(
        y_train_actual,
        y_train_pred,
        y_test_actual,
        y_test_pred,
        y[-time_step:] * max_price,
        future_x_data * max_price,
        time_step,
    )
