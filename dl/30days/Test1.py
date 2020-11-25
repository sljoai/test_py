import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dftrain_raw = pd.read_csv('data/titanic/train.csv')
dftest_raw = pd.read_csv('data/titanic/test.csv')
dftrain_raw.head(10)


# %matplotlib inline
# config InlineBackend.figure_format = 'png'
# 可视化数据
## label 分布情况
# ax = dftrain_raw['Survived'].value_counts().plot(kind='bar', figsize=(12, 8), fontsize=15, rot=0)
# ax.set_ylabel('Counts', fontsize=15)
# ax.set_xlabel('Survived', fontsize=15)
# plt.show()

## 年龄分布
# ax = dftrain_raw['Age'].plot(kind='hist', bins=20, color='purple', figsize=(12, 8), fontsize=15)
# ax.set_xlabel('Frequency', fontsize=15)
# ax.set_ylabel('Age', fontsize=15)
# plt.show()

## 年龄 和 label 的相关性
# ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density',
#                                                     figsize=(12, 8), fontsize=15)
# dftrain_raw.query('Survived==1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
# ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
# ax.set_ylabel('Density', fontsize=15)
# ax.set_xlabel('Age', fontsize=15)
# plt.show()

# Survived:0代表死亡，1代表存活【y标签】
# Pclass:乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
# Name:乘客姓名 【舍去】
# Sex:乘客性别 【转换成bool特征】
# Age:乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
# SibSp:乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
# Parch:乘客父母/孩子的个数(整数值)【数值特征】
# Ticket:票号(字符串)【舍去】
# Fare:乘客所持票的价格(浮点数，0-500不等) 【数值特征】
# Cabin:乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
# Embarked:乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】
# 数据预处理
def preprocessing(dfdata):
    dfresult = pd.DataFrame()

    # Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)

    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp, Parch, Face
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Cabin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)


x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape=", x_train.shape)
print('y_train.shape=', y_train.shape)

# 定义模型
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# 训练模型
model.compile(optimizer='adam',  # 含义是什么呢
              loss='binary_crossentropy',
              metrics=['AUC'])

history_result = model.fit(x_train, y_train,
                           batch_size=64,
                           epochs=30,
                           validation_split=0.2,  # 分割一部分训练数据用于验证
                           )


# 评估模型

# 输出模型评估参数
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


# plot_metric(history_result, "loss")
# plot_metric(history_result, "AUC")

## 使用测试集评估模型
evaluate_result = model.evaluate(x=x_test, y=y_test, verbose=1)
print(evaluate_result)

# 使用模型
## 预测类别
results = model.predict_classes(x_test[0:10], verbose=1)
print(results)

# 保存模型
keras_model_path = "./data/titanic/keras_model.h5"
## 以 keras 方式保存现有模型
### 方式一
# model.save(keras_model_path)
#
# del model
# model = models.load_model(keras_model_path) ## 加载模型
# reevaluate_results = model.evaluate(x_test, y_test)
# print(reevaluate_results)

### 方式二
# json_str = model.to_json()  ## 保存模型结构
# keras_model_weight_path = "./data/titanic/keras_model_weight.h5"
# model.save_weights(keras_model_weight_path)  ## 保存模型权重
# model_json = models.model_from_json(json_str)  ## 恢复模型
# model_json.compile(optimizer='adam',
#                    loss='binary_crossentropy',
#                    metrics=['AUC'])
# model_json.load_weights(keras_model_weight_path)
# reevaluate_results = model_json.evaluate(x_test, y_test)
# print(reevaluate_results)

# TensorFlow 原生方式保存
tensorflow_model_weight_path = "./data/titanic/tf_model_weights.ckpt"
model.save_weights(tensorflow_model_weight_path, save_format="tf")  ## 保存权重张量
tensorflow_model_path = "./data/titanic/tf_model_savemodel"
model.save(tensorflow_model_path, save_format="tf")  ## 保存模型结构与模型参数文件，便于跨平台部署
model_load = tf.keras.models.load_model(tensorflow_model_path)  ## 加载模型
reevaluate_results = model_load.evaluate(x_test, y_test)
print(reevaluate_results)
