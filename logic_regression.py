import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ======================
# 边界函数推导
# ======================
# 一、一阶边界函数（线性边界）
# 1. 基本原理：逻辑回归的预测概率
#    P(y=1|x) = 1 / (1 + e^-(w^T x + b))
#    其中 w 是权重向量，b 是偏置项，x 是输入特征向量
# 2. 边界条件：当预测概率 P(y=1|x) = 0.5 时，是正类和负类的分界点
#    1 / (1 + e^-(w^T x + b)) = 0.5
#    - 整理得：w^T x + b = 0
# 3. 二维特征的具体形式（x = [x1, x2]）：
#    w1*x1 + w2*x2 + b = 0  # 线性决策边界（直线）

# 二、二阶边界函数（非线性边界）
# 1. 引入二阶特征：构造新的特征向量
#    x' = [x1, x2, x1², x2², x1x2]  # 包含原特征的二阶项
# 2. 逻辑回归模型：使用新特征向量
#    P(y=1|x') = 1 / (1 + e^-(w'^T x' + b'))
#    其中 w' 是新的权重向量，b' 是偏置项
# 3. 边界条件：当 P(y=1|x') = 0.5 时
#    w'^T x' + b' = 0
# 4. 二维特征的具体形式：
#    w1'*x1 + w2'*x2 + w3'*x1² + w4'*x2² + w5'*x1x2 + b' = 0
#    这是一个二次方程，对应椭圆、双曲线或抛物线等二次曲线
# 三、总结
# - 一阶边界：线性决策边界，适用于线性可分数据
# - 二阶边界：非线性决策边界，通过引入二次特征，适用于非线性可分数据
# - 推导核心：均基于逻辑回归中概率为0.5时的条件，即线性组合为0的方程


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.loc[:, ["Exam1", "Exam2"]]
    y = data.loc[:, "Pass"]
    return data, X, y


def show_origin_data(data):
    mask = data.loc[:, "Pass"] == 1
    plt.figure()
    paxx = plt.scatter(data.loc[:, "Exam1"][mask], data.loc[:, "Exam2"][mask])
    fail = plt.scatter(data.loc[:, "Exam1"][~mask], data.loc[:, "Exam2"][~mask])
    plt.xlabel("Exam1 Score")
    plt.ylabel("Exam2 Score")
    plt.title("Exam1-Exam2")
    plt.legend((paxx, fail), ("Pass", "Fail"))
    plt.show()


def create_second_order_features(X):
    """
    构造二阶特征

    Args:
        X: 原始特征矩阵（包含Exam1和Exam2）

    Returns:
        X_new: 包含二阶特征的新特征矩阵
    """
    x1 = X.loc[:, "Exam1"]
    x2 = X.loc[:, "Exam2"]
    x1_2 = x1 * x1
    x2_2 = x2 * x2
    x1x2 = x1 * x2
    X_new = {"X1": x1, "X2": x2, "X1_2": x1_2, "X2_2": x2_2, "X1X2": x1x2}
    X_new = pd.DataFrame(X_new)
    return X_new


def create_test_data(first_order=False, **kwargs):
    """
    创建测试数据

    Args:
        first_order: 是否为一阶特征
        **kwargs: 测试数据参数
            exam1: Exam1分数
            exam2: Exam2分数

    Returns:
        x_test: 测试数据
    """
    exam1 = kwargs.get("exam1", 80)
    exam2 = kwargs.get("exam2", 60)

    if first_order:
        # 一阶特征测试数据
        x_test = pd.DataFrame([[exam1, exam2]], columns=["Exam1", "Exam2"])
    else:
        # 二阶特征测试数据
        x_test = pd.DataFrame(
            [[exam1, exam2, exam1**2, exam2**2, exam1 * exam2]],
            columns=["X1", "X2", "X1_2", "X2_2", "X1X2"],
        )

    return x_test


def plot_first_order_boundary(data, model):
    """
    绘制一阶决策边界

    Args:
        data: 数据
        model: 训练好的一阶模型
    """
    x1 = data.loc[:, "Exam1"]
    theta0 = model.intercept_
    theta1, theta2 = model.coef_[0][0], model.coef_[0][1]

    # 一阶边界函数 theta0 + theta1 * x1 + theta2 * x2 = 0
    x2_new = -(theta0 + theta1 * x1) / theta2

    plt.figure()
    plt.plot(x1, x2_new, color="red")
    mask = data.loc[:, "Pass"] == 1
    paxx = plt.scatter(data.loc[:, "Exam1"][mask], data.loc[:, "Exam2"][mask])
    fail = plt.scatter(data.loc[:, "Exam1"][~mask], data.loc[:, "Exam2"][~mask])
    plt.xlabel("Exam1 Score")
    plt.ylabel("Exam2 Score")
    plt.title("First Order Decision Boundary")
    plt.legend((paxx, fail), ("Pass", "Fail"))
    plt.show()


def plot_second_order_boundary(data, model):
    """
    绘制二阶决策边界

    Args:
        data: 数据
        model: 训练好的二阶模型
    """
    x1 = data.loc[:, "Exam1"]
    theta0 = model.intercept_
    theta1, theta2, theta3, theta4, theta5 = (
        model.coef_[0][0],
        model.coef_[0][1],
        model.coef_[0][2],
        model.coef_[0][3],
        model.coef_[0][4],
    )

    # 二阶边界函数 theta0 + theta1 * x1 + theta2 * x2 + theta3 * x1^2 + theta4 * x2^2 + theta5 * x1x2 = 0
    x1_new = x1.sort_values()
    a = theta4
    b = theta2 + theta5 * x1_new
    c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new

    # 一元二次方程求根公式，确保判别式非负
    discriminant = b * b - 4 * a * c
    # 只保留判别式非负的点
    valid_indices = discriminant >= 0
    x1_valid = x1_new[valid_indices]
    if len(x1_valid) > 0:
        sqrt_discriminant = np.sqrt(discriminant[valid_indices])
        x2_new = (-b[valid_indices] + sqrt_discriminant) / (2 * a)
    else:
        # 如果没有有效点，创建空数组
        x2_new = np.array([])

    plt.figure()
    if len(x1_valid) > 0:
        plt.plot(x1_valid, x2_new, color="red")
    mask = data.loc[:, "Pass"] == 1
    paxx = plt.scatter(data.loc[:, "Exam1"][mask], data.loc[:, "Exam2"][mask])
    fail = plt.scatter(data.loc[:, "Exam1"][~mask], data.loc[:, "Exam2"][~mask])
    plt.xlabel("Exam1 Score")
    plt.ylabel("Exam2 Score")
    plt.title("Second Order Decision Boundary")
    plt.legend((paxx, fail), ("Pass", "Fail"))
    plt.show()


def test_first_order_model(model, exam1=80, exam2=60):
    """
    测试一阶模型

    Args:
        model: 训练好的一阶模型
        exam1: Exam1分数
        exam2: Exam2分数

    Returns:
        result: 测试结果（"Pass" 或 "Fail"）
    """
    x_test = create_test_data(first_order=True, exam1=exam1, exam2=exam2)
    x_test_result = model.predict(x_test)
    result = "Pass" if x_test_result[0] == 1 else "Fail"
    return result


def test_second_order_model(model, exam1=80, exam2=60):
    """
    测试二阶模型

    Args:
        model: 训练好的二阶模型
        exam1: Exam1分数
        exam2: Exam2分数

    Returns:
        result: 测试结果（"Pass" 或 "Fail"）
    """
    x_test = create_test_data(first_order=False, exam1=exam1, exam2=exam2)
    x_test_result = model.predict(x_test)
    result = "Pass" if x_test_result[0] == 1 else "Fail"
    return result


def main():
    """
    主函数
    """
    # 加载数据
    data, X, y = load_data("data/examdata.csv")
    show_origin_data(data)
    # 训练一阶模型
    print("=== First Order Model ===")
    model_first = LogisticRegression(max_iter=1000)
    model_first.fit(X, y)
    y_pred_first = model_first.predict(X)
    acc_first = accuracy_score(y, y_pred_first)
    print(f"Accuracy: {acc_first}")

    # 测试一阶模型
    result_first = test_first_order_model(model_first, exam1=65, exam2=60)
    print(f"Test result: {result_first}")

    # 绘制一阶边界
    plot_first_order_boundary(data, model_first)

    # 构造二阶特征
    X_second = create_second_order_features(X)

    # 训练二阶模型
    print("\n=== Second Order Model ===")
    model_second = LogisticRegression(max_iter=1000)
    model_second.fit(X_second, y)
    y_pred_second = model_second.predict(X_second)
    acc_second = accuracy_score(y, y_pred_second)
    print(f"Accuracy: {acc_second}")

    # 测试二阶模型
    result_second = test_second_order_model(model_second, exam1=60, exam2=65)
    print(f"Test result: {result_second}")

    # 绘制二阶边界
    plot_second_order_boundary(data, model_second)


if __name__ == "__main__":
    main()
