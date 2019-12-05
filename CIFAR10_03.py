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


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, input, training=None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])
        # resblock
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # there are [b,512,h,w]
        # 自适应
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, input, training=None):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [b,c]
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))
        # just down sample one time
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[0:1000]
x_test = x_test[0:200]
y_train = y_train[0:1000]
y_test = y_test[0:200]
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(1000).map(preprocess).batch(10)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.map(preprocess).batch(10)

sample = next(iter(train_data))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)
    for epoch in range(50):
        for step, (x, y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 10 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss))
        total_num = 0
        total_correct = 0
        for x, y in test_data:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            preb = tf.argmax(prob, axis=1)
            preb = tf.cast(preb, dtype=tf.int32)
            correct = tf.cast(tf.equal(preb, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print('epoch:',epoch, 'accuracy:', acc)


if __name__ == '__main__':
    main()
