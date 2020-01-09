#has bug
import numpy as np
# import pandas as pd
import os, shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
#from keras import optimizers, losses
from tensorflow.keras.datasets import cifar10
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
nb_classes = 10
epochs = 12

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = tf.squeeze(y_train, axis=1)
# y_test = tf.squeeze(y_test, axis=1)
#y_train = np_utils.to_categorical(y_train, 100)
#y_test = np_utils.to_categorical(y_test, 100)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = 2 * x_train / 255. - 1
x_test = 2 * x_test / 255. - 1
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
# x_train = tf.cast(x_train, dtype=tf.float32) / 255.
# x_test = tf.cast(x_test, dtype=tf.float32) / 255.
# y_train=tf.cast(y_train,dtype=tf.int32)
# y_test=tf.cast(y_test,dtype=tf.int32)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

network = Sequential(
    [
        Convolution2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu, input_shape=(32, 32, 3)),
        Convolution2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        Convolution2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        Convolution2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        Convolution2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        Convolution2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        Convolution2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        Convolution2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        Convolution2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        Convolution2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        Flatten(),
        Dense(256, activation=tf.nn.relu),
        Dense(128, activation=tf.nn.relu),
        Dense(100, activation=None)
    ]
)
network.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(lr=1e-4),
                metrics=['accuracy'])
network.build(input_shape=[None, 32, 32, 3])
network.summary()
network.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(x_test, y_test))
# network.fit(x_train, y_train,
#             verbose=1, validation_data=(x_test, y_test))
score = network.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
