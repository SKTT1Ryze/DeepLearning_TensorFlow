# import pandas as pd
import os
import tensorflow as tf
import tensorflow_core
from tensorflow_core import keras
from tensorflow_core.keras import layers
from tensorflow_core.keras import Sequential
from tensorflow_core.keras import optimizers, losses
# import keras
# from keras import optimizers, losses, metrics, layers
from keras.datasets import cifar10
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def progress(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).map(progress).batch(128)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(progress).batch(64)

train_next = next(iter(db_test))
print(train_next[0].shape, train_next[1].shape)

cov_network = Sequential(
    [
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same')
    ]
)

# class myDense(layers.Layer):
#     def __init__(self, in_dim, out_dim):
#         super(myDense, self).__init__()
#         self.kernel = self.add_variable('w', [in_dim, out_dim])
#         self.bias = self.add_variable('b', out_dim)
#
#     def call(self, inputs, training=None):
#         out = inputs @ self.kernel + self.bias
#         return out
#
#
# class myModel(keras.Model):
#     def __init__(self):
#         super(myModel, self).__init__()
#         self.fc1 = myDense(512, 256)
#         self.fc2 = myDense(256, 128)
#         self.fc3 = myDense(128, 100)
#
#     def call(self, inputs, training=None):
#         x = self.fc1(inputs)
#         out = tf.nn.relu(x)
#         x = self.fc2(out)
#         out = tf.nn.relu(x)
#         x = self.fc3(out)
#         return x
#

fc_network = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(100, activation=None),
])


def main():
    cov_network.build(input_shape=[None, 32, 32, 3])
    fc_network.build(input_shape=[None, 512])
    all_trainable = cov_network.trainable_variables + fc_network.trainable_variables

    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                out = cov_network(x)
                out = tf.reshape(out, [-1, 512])
                logits = fc_network(out)

                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, all_trainable)
            optimizer.apply_gradients(zip(grad, all_trainable))
            if step % 100 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss))
        total_num = 0
        total_correct = 0
        for x, y in db_test:
            out = cov_network(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_network(out)
            prob = tf.nn.softmax(logits, axis=1)
            pre = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            total_correct += tf.reduce_sum(tf.cast(tf.equal(pre, y), dtype=tf.int32))
            total_num += x.shape[0]
        last_correct = int(total_correct) / total_num
        print("epoch:", epoch, last_correct)


if __name__ == '__main__':
    main()
