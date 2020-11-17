import tensorflow as tf

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 载入MNIST数据集
mnist = tf.keras.datasets.mnist
# 划分训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定义模型结构和模型参数
model = tf.keras.models.Sequential([
    # 输入层 28 * 28 维矩阵
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 128维隐层，使用relu作为激活函数
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # 输出层采用 softmax 模型，处理多分类问题
    tf.keras.layers.Dense(10, activation='softmax')
])

# 定义模型优化方法（adam）,损失函数（sparse_categorical_crossentropy）和评估指标（accuracy）
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，进行5轮迭代更新（epochs=5）
model.fit(x_train, y_train, epochs=5)
# 评估模型
model.evaluate(x_test, y_test, verbose=2)
