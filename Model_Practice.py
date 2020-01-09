import numpy as np
import os
import tensorflow as tf
import tensorflow_core
from tensorflow_core import keras
from tensorflow_core.keras import layers
from tensorflow_core.keras import Sequential
from tensorflow_core.keras import optimizers, losses
from sklearn.datasets import make_moons
from sklearn.datasets import load_sample_images
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N_SAMPLES = 2000  # 采样点数
TEST_SIZE = 0.3  # 测试数量比率
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
Y_train = []
print(X_train.shape)
print(y_train.shape)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
network = Sequential([layers.Dense(12, activation='relu', input_dim=2),
                      layers.Dense(20, activation='relu'),
                      layers.Dense(25, activation='relu'),
                      layers.Dense(15, activation='relu'),
                      layers.Dense(2, activation='sigmoid')])
network.build(input_shape=(None, 2))
network.summary()
network.compile(optimizer=optimizers.Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 返回训练信息保存在 history 中
history = network.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), validation_freq=5)
# history = network.fit(X_train,Y_train)
# print(history)

# imageset = load_sample_images()
# image_train_x = np.array([imageset.images[0], imageset.images[1]])
# print(image_train_x.shape)
# print(image_train_x)
# # print(imageset.images)
# # print(imageset.images[0].shape)
# # for i in range(0,2):
# #     image_train_x.append(imageset.images[i])
# # print(image_train_x)
