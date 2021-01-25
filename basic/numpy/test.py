import numpy as np



def sigmoid(x):
    """
    定义sigmoid函数
    :param x: 待转换的数据，向量或矩阵
    :return: 转换后的结果
    """
    return 1 / (1 + np.exp(-x))


# 构造一个两层神经网络 2*4*3
# 第一层权重
w1 = np.random.randn(2, 4)
# 第一层偏置
b1 = np.random.randn(4)
# 第一层输入
x = np.random.randn(10, 2)
# 隐藏层输出
h = np.dot(x, w1) + b1
# 使用激活函数进行非线性转换
a = sigmoid(h)

# 第二层权重
w2 = np.random.randn(4, 3)
# 第二层偏置
b2 = np.random.randn(3)
s = np.dot(a, w2) + b2
