import numpy as np


class Network(object):
    def __init__(self, sizes):
        # sizes 表示神经网络中每层的神经元数量。如 [2, 3 ,1]
        #   L1      L2      L3
        #  ----------------------
        #           N3
        #   N1
        #           N4      N6
        #   N2
        #           N5
        # 第一层是输入层, 最后一层是输出层, 中间是隐藏层
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 为除第一层外的每一层的每一个神经元随机生成一个偏置项
        # biases[0] 是第二层的神经元的偏置项, biases[1] 是第三层的偏置项, 以此类推
        # [
        #  [ 第二层
        #   [0.57155959]  第一个神经元的偏置项
        #   [0.29725612]  第二个神经元的偏置项
        #   [1.19600182]  第三个神经元的偏置项
        #  ],
        #  [ 第三层
        #   [-0.17062043] 第一个神经元的偏置项
        #  ]
        # ]
        # 优化点: np.random.randn(y, 1)可能产生较大值,导致梯度爆炸, 1/100
        self.biases = [np.random.randn(y, 1) * 0.01 for y in sizes[1:]]

        # 为每一层生成一个权重矩阵，维度为 (当前层神经元数量, 前一层神经元数量)，表示该层每个神经元与前一层每个神经元之间的权重
        # [
        #  [
        #   [ 1.47794118  0.29770183] N3 与 N1、N2 的连接权重
        #   [-0.67723849  0.13104258] N4 与 N1、N2 的连接权重
        #   [-0.13024933 -0.53765215] N5 与 N1、N2 的连接权重
        #   ],
        #  [
        #   [-0.17062043  0.57155959  0.29725612] N6 与 N3、N4、N5 的连接权重
        #  ]
        # ]
        # 综上:
        # N3的输出 = Out(N1) × 1.47794118 + Out(N2) × 0.29770183 + 0.57155959
        # N6的输出 = Out(N3) × -0.17062043 + Out(N4) × 0.57155959 + Out(N5) × 0.29725612 + (-0.17062043)（N6的偏置项）
        # Out(Nx) 表示神经元 Nx 的输出
        self.weights = [
            np.random.randn(y, x) * 0.01 for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def sigmoid(self, z):
        # 防止溢出：clip z 到合理范围
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # 1 / (1 + e^-z) 的导数
    # = (1 / (1 + e^-z)) * (1 - (1 / (1 + e^-z)))
    # = sigmoid(z) * (1 - sigmoid(z))
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, x):
        # 即 wx+b
        # 该函数接受一个输入向量 x，并返回网络的输出
        for b, w in zip(self.biases, self.weights):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def evaluate(self, test_data):
        # 评估网络在测试数据上的表现
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # 用随机梯度下降算法来训练神经网络
        # training_data 是一个包含训练样本的列表，每个样本是一个元组 (x, y)，其中 x 是输入向量，y 是对应的输出向量
        # epochs 是训练的轮数
        # mini_batch_size 是每个小批量的大小，
        # eta 是学习率

        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for batch in batches:
                self.update_mini_batch(batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, batch, eta):
        # 更新网络的权重和偏置项
        # batch 是一个包含训练样本的列表，每个样本是一个元组 (x, y)，其中 x 是输入向量，y 是对应的输出向量
        # eta 是学习率
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            dbs = [db + b for db, b in zip(dbs, delta_b)]
            dws = [dw + w for dw, w in zip(dws, delta_w)]

        # w = w - eta * dw
        self.weights = [w - (eta / len(batch)) * dw for w, dw in zip(self.weights, dws)]
        self.biases = [b - (eta / len(batch)) * db for b, db in zip(self.biases, dbs)]

    def backprop(self, x, y):
        # x 是输入向量，y 是对应的输出向量

        # 反向传播算法的核心就是计算每一层的误差 delta, 然后根据 delta 来计算每一层的权重和偏置项的梯度
        # 反向传播算法的步骤如下:
        # 1. 前向传播: 计算每一层的 z 向量和激活值 a 向量
        # 2. 反向传播: 从输出层开始, 逐层向前计算每一层的误差 delta, 然后根据 delta 来计算每一层的权重和偏置项的梯度

        # y 是期望的输出向量
        # z = w*x + b
        # a = f(z)
        # L = 1/2(y - a)^2 MSE 均方误差损失函数
        # 因此损失函数对偏置的b的导数为:
        # dL/db = dL/da * da/dz * dz/db
        # dL/da = a - y
        # da/dz = f'(z)
        # dz/db = 1
        # dz/dw = x
        # 因此 dL/db = (a - y) * f'(z)
        # 同理 dL/dw = dL/da * da/dz * dz/dw = (a - y) * f'(z) * x = dL/db * x
        # 误差 delta = dL/dz = dL/da * da/dz = (a - y) * f'(z)
        # 所以 dL/db = delta, dL/dw = delta * x
        #
        # 推广到每一层：
        # a_l 是 l+1层的输入, 也是l层的输出。
        # delta_l = dL/dz = dL/da_l * da_l/dz = dL/da_l * f'(z_l)
        # a_l 会影响一层的所有z_(l+1)
        # 又 每个a_l_i : z_(l+1)_i = w_(l+1)_i * a_l_i + b_(l+1)_i
        # 由多元链式法则可得:
        # dL/da_l = sum(dL/dz_(l+1) * dz_(l+1)/da_l_i) = sum(delta_(l+1) * w_(l+1)_i)
        # 写为向量形式:
        # dL/da_l = w_(l+1).T * delta_(l+1)
        # 因此 delta_l = (w_(l+1).T * delta_(l+1)) * f'(z_l)

        # 它把后一层的误差信号delta沿着网络反向传播,
        # 通过权重矩阵的转置“分配”给当前层的每个神经元，反映了当前层激活值对后一层所有神经元误差的贡献总和

        zs = []
        activations = [x]  # 存储每层的激活值
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b  # 该层的 z 向量
            x = self.sigmoid(z)  # 该层的激活值也是下一层的输入
            zs.append(z)  # 存储每层的 z 向量（即 wx+b）
            activations.append(x)  # 存储每层的激活值

        # 反向传播
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]
        # b 的梯度 = (a - y) * f'(z)
        dbs[-1] = delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
        # 前一层的激活值就是当前层的输入, 因此w的梯度 = 偏置项的梯度 * 前一层的激活值
        dws[-1] = np.dot(delta, activations[-2].transpose())

        # 从网络的倒数第二层开始, 逐层向前计算每一层的误差和梯度
        for l in range(2, self.num_layers):
            # 该层的误差 = 后一层的误差 * 后一层的权重矩阵的转置 * 当前层的激活函数的导数
            delta = np.dot(
                self.weights[-l + 1].transpose(), delta
            ) * self.sigmoid_prime(zs[-l])
            dbs[-l] = delta
            dws[-l] = np.dot(delta, activations[-l - 1].transpose())

        return dbs, dws


def load_scaled_mnist_data():
    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    tr_d = (X_train[:50000], y_train[:50000])
    va_d = (X_train[50000:], y_train[50000:])
    te_d = (X_test, y_test)

    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    # 归一化, 否则影响训练稳定性
    training_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) / 255.0 for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    print(len(training_data), len(test_data))  # 50000 10000
    print(training_data[0][0].shape, training_data[0][1].shape)  # (784, 1) (10, 1)
    print(test_data[0][0].shape, test_data[0][1].shape)  # (784, 1) ()

    return training_data, validation_data, test_data


if __name__ == "__main__":
    training_data, validation_data, test_data = load_scaled_mnist_data()
    net = Network([784, 30, 10])
    net.sgd(training_data, epochs=30, mini_batch_size=32, eta=0.1, test_data=test_data)
    # Epoch 0: 1135 / 10000
    # Epoch 1: 3487 / 10000
    # Epoch 2: 6438 / 10000
    # Epoch 3: 7915 / 10000
    # Epoch 4: 8634 / 10000
    # Epoch 5: 8901 / 10000
    # ....
    # Epoch 25: 9324 / 10000
    # Epoch 26: 9332 / 10000
    # Epoch 27: 9336 / 10000
    # Epoch 28: 9337 / 10000
    # Epoch 29: 9350 / 10000
