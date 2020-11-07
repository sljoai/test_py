import numpy as np

X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

X_array = np.array(X)
Y_array = np.array(Y)

print(X_array)
# 转换为5*1矩阵
print(Y_array.reshape(5, 1))
# 矩阵加法
print(X + Y)

# 矩阵点乘
res_tmp = np.dot(X_array, Y_array)
print(res_tmp)

res_tmp1 = np.sum(X_array * Y_array)
print(res_tmp1)

# 向量之间的乘法

# 向量的叉乘

# 对应项相乘
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
# 点乘
print(np.inner(a, b))
# 叉乘
print(np.outer(a, b))
# 对应项相乘
print(np.multiply(a, b))

# TODO: 语法结构待理解
c = np.mat(((1, 2), (5, 6)))
d = np.mat(((0, 1), (2, 3)))
print(c)
print(c + d)
