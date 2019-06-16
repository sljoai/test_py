import numpy as np

X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

X_array = np.array(X)
Y_array = np.array(Y)

print(X_array)
# 转换为5*1矩阵
print(Y_array.reshape(5, 1))

# 矩阵点成
res_tmp = np.dot(X_array, Y_array)
print(res_tmp)

res_tmp1 = np.sum(X_array * Y_array)
print(res_tmp1)
