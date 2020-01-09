#has bug
import numpy as np
# import pandas as pd
import os, shutil
import tensorflow as tf
# import tensorflow_core
# from tensorflow_core import keras
# from tensorflow_core.keras import layers
# from tensorflow_core.keras import Sequential
# from tensorflow_core.keras import optimizers, losses
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
nb_classes = 10
epochs = 12

img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# model = Sequential()
# model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
#                         padding='same',
#                         input_shape=input_shape))  # 卷积层1
# model.add(Activation('relu'))  # 激活层
# model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
# model.add(Activation('relu'))  # 激活层
# model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
# model.add(Dropout(0.25))  # 神经元随机失活
# model.add(Flatten())  # 拉成一维数据
# model.add(Dense(128))  # 全连接层1
# model.add(Activation('relu'))  # 激活层
# model.add(Dropout(0.5))  # 随机失活
# model.add(Dense(nb_classes))  # 全连接层2
# model.add(Activation('softmax'))  # Softmax评分
#
model = Sequential([
    Convolution2D(6, kernel_size=3, strides=1, input_shape=input_shape),  # 第一个卷积层，6个3*3卷积核
    Activation('relu'),  # 激活函数
    MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    Activation('relu'),  # 激活函数
    Convolution2D(16, kernel_size=3, strides=1),  # 第二个卷积层,16个3*3卷积核
    Activation('relu'),  # 激活函数
    MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    Activation('relu'),  # 激活函数
    Flatten(),  # 打平层

    Dense(120, activation='relu'),  # 全连接层，120个节点
    Activation('relu'),  # 激活函数
    Dense(84, activation='relu'),  # 全连接层，84个节点
    Activation('relu'),  # 激活函数
    Dense(10),  # 全连接层，10个节点
    Activation('softmax')  # 激活函数
])

# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.build(input_shape=(4, img_rows, img_cols, 1))
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
