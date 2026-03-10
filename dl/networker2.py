import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    zx = sigmoid(z)
    return zx * (1 - zx)  # 减少一次计算
    # return sigmoid(z) * (1 - sigmoid(z))


class QuadraticCost(object):
    # 平方误差损失函数, 适用于回归问题
    @staticmethod
    def fn(a, y):
        # C = 1/2 * (a - y)^2
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        # 计算输出层的误差项 delta
        # delta = dC/dz = dC/da * da/dz
        # dC/da = a - y
        # L层的输出 a = f(z)
        # da/dz = df(z)/dz = f'(z)（f 即 sigmoid 函数）
        # 因此 delta = (a - y) * f'(z)
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        # 计算单个样本的交叉熵损失
        # 公式：C = -y * log(a) - (1 - y) * log(1 - a)
        # 其中 a 是网络输出层的激活值（经过 sigmoid 后的概率）
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        # 计算输出层的误差项 delta
        # delta = dC/dz = dC/da * da/dz
        # dC/da = -y/a - (1-y) * (-1)/(1-a) = (a - y) / (a * (1 - a))
        # L层的输出 a = f(z)
        # da/dz = df(z)/dz = f'(z) = f(z)*[1 - f(z)] = a(1 - a) （f 即 sigmoid 函数）
        # 因此 delta = [(a - y) / (a*(1-a))] * a*(1-a) = a - y
        return a - y


class Network2(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weights_initializer()
        self.cost = cost

    def default_weights_initializer(self):
        # 形状为(y , 1)的数组,每个元素都服从标准正态分布(均值为0,标准差为1)
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # 形状为(y, x),均值为0 标准差为 1/sqrt(x) 的数组, x 是前一层的神经元数量, y 是当前层的神经元数量
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def sgd(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        evaluation_data=None,
    ):
        if evaluation_data:
            n_evaluation = len(evaluation_data)

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for batch in batches:
                self.update_mini_batch(batch, eta, lmbda, len(training_data))

            if evaluation_data is not None:
                evaluation_accuracy = self.accuracy(evaluation_data)
                print(f"Epoch {j}, accuracy: {evaluation_accuracy} / {n_evaluation}")

    def update_mini_batch(self, batch, eta, lmbda, n):
        """
        lmbda 是正则化参数, n 是训练数据的总数量
        这两个参数用于实现 L2 正则化, 以防止过拟合
        L2 正则化通过在损失函数中添加一个项来惩罚权重的大小
        """
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            dbs = [db + b for db, b in zip(dbs, delta_b)]
            dws = [dw + w for dw, w in zip(dws, delta_w)]

        self.weights = [
            # L2 正则化的权重更新公式:
            # w = (1 - eta * (lmbda / n)) * w - (eta / len(batch)) * dw
            (1 - eta * (lmbda / n)) * w - (eta / len(batch)) * dw
            for w, dw in zip(self.weights, dws)
        ]
        self.biases = [b - (eta / len(batch)) * db for b, db in zip(self.biases, dbs)]

    def backprop(self, x, y):
        zs = []
        activations = [x]
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            x = sigmoid(z)  # 该层的激活值也是下一层的输入
            zs.append(z)
            activations.append(x)

        # 反向传播
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]

        dbs[-1] = delta = self.cost.delta(zs[-1], activations[-1], y)
        dws[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            dbs[-l] = delta = np.dot(
                self.weights[-l + 1].transpose(), delta
            ) * sigmoid_prime(zs[-l])
            dws[-l] = np.dot(delta, activations[-l - 1].transpose())

        return dbs, dws

    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


if __name__ == "__main__":
    import networker

    training_data, validation_data, test_data = networker.load_scaled_mnist_data()

    w2 = Network2([784, 30, 10])
    w2.sgd(
        training_data,
        epochs=30,
        mini_batch_size=32,
        eta=3.0,
        lmbda=10.0,
        evaluation_data=validation_data,
    )

    # Epoch 0, accuracy: 8893 / 10000
    # Epoch 1, accuracy: 9351 / 10000
    # Epoch 2, accuracy: 8800 / 10000
    # Epoch 3, accuracy: 9380 / 10000
    # Epoch 4, accuracy: 9443 / 10000
