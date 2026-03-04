# mlp 多分类
# 基于图像数字的自动识别

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


def reshape_sample():
    raw = [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
    ]
    print(raw)
    print(np.array(raw).reshape(2, 9))


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # (60000, 28, 28) (60000,)
    # 60000张图片, 每个图片28*28=784个像素点
    print(X_train.shape, y_train.shape)
    img1 = X_train[0]
    feature_size = img1.shape[0] * img1.shape[1]
    # 28*28=784个像素点
    print(img1.shape[0], img1.shape[1], feature_size)
    # 60000 * 784
    # 每个图片的像素点展开为一个784维的向量
    X_train_format = X_train.reshape(X_train.shape[0], feature_size)
    X_test_format = X_test.reshape(X_test.shape[0], feature_size)
    print(X_train_format.shape, X_test_format.shape)
    # 归一化, 每个像素点的值除以255, 范围变为[0, 1]
    X_train_normal = X_train_format / 255
    X_test_normal = X_test_format / 255
    # 对标签进行one-hot编码 即每个标签转换为一个10维的向量, 只有对应位置为1, 其他位置为0
    y_train_format = to_categorical(y_train)
    y_test_format = to_categorical(y_test)
    print(y_train[0], y_test[1])
    print(y_train_format[0], y_test_format[1])

    # 建立mlp模型
    mlp = Sequential()
    mlp.add(Input(shape=(feature_size,)))
    mlp.add(Dense(units=feature_size // 2, activation="sigmoid"))
    mlp.add(Dense(units=feature_size // 2, activation="sigmoid"))
    mlp.add(Dense(units=10, activation="softmax"))  # 多分类
    mlp.summary()
    # 多分类交叉熵损失函数, adam优化器
    mlp.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01))

    mlp.fit(
        X_train_normal,
        y_train_format,
        epochs=20,
        verbose=1,
        batch_size=256,
    )

    y_train_pred = (mlp.predict(X_train_normal) > 0.5).astype("int32")
    print(y_train_pred[0])
    accuracy_train = accuracy_score(y_train_format, y_train_pred)
    print("train acc: ", accuracy_train)
    y_test_pred = (mlp.predict(X_test_normal) > 0.5).astype("int32")
    print(y_test_pred[0])
    accuracy_test = accuracy_score(y_test_format, y_test_pred)
    print("test acc: ", accuracy_test)

    # 随机抽取十张图片查看训练结果
    random_index = np.random.randint(0, X_test.shape[0], 10)
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[random_index[i]])
        plt.title(f"({np.argmax(y_test_pred[random_index[i]])}, {y_test[random_index[i]]})")  
        plt.axis("off")
    plt.show()
