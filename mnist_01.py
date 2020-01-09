#no bug
import numpy as np
import pandas as pd
import os, shutil
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=10)
print(x_train.shape, y_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)

# networks = Sequential([
#     layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层，6个3*3卷积核
#     layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
#     layers.ReLU(),  # 激活函数
#     layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层,16g个3*3卷积核
#     layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
#     layers.ReLU(),  # 激活函数
#     layers.Flatten(),  # 打平层
#
#     layers.Dense(120, activation='relu'),  # 全连接层，120个节点
#     layers.Dense(84, activation='relu'),  # 全连接层，84个节点
#     layers.Dense(10)  # 全连接层，10个节点
# ])

# networks.build(input_shape=(4, 28, 28, 1))  # build
# networks.summary()
# criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=0.001)

networks = keras.Sequential([layers.Dense(784, activation='relu'),
                             layers.Dense(256, activation='relu'),
                             layers.Dense(10)])


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = networks(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        grad = tape.gradient(loss, networks.trainable_variables)
        optimizer.apply_gradients(zip(grad, networks.trainable_variables))
        if step % 100 == 0:
            print("数据集迭代第%s次，训练step为%s，loss函数为：%s" % (epoch, step, loss.numpy()))


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
