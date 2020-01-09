# coding: utf-8

# #### 通过RNN使用imdb数据集完成情感分类任务
#no bug
from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import matplotlib.pyplot as plt

tf.__version__

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_LOG_LEVEL'] = '2'

# 超参数
vocab_size = 10000
max_review_length = 80
embedding_dim = 100
units = 64
num_classes = 2
batch_size = 32
epochs = 10

# 加载数据集
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# 建立词典
word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNSED>"] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reversed_word_index.get(i, '?') for i in text])


train_data = train_data[:20000]
val_data = train_data[20000:25000]

train_labels = train_labels[:20000]
val_labels = train_labels[20000:25000]

# 补齐数据
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post',
                                                        maxlen=max_review_length)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post',
                                                       maxlen=max_review_length)


# 构建模型
class RNNModel(keras.Model):

    # def __init__(self, units, num_classes, num_layers):
    #     super(RNNModel, self).__init__()
    #
    #     self.units = units
    #
    #     self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_review_length)
    #     """
    #     self.lstm = keras.layers.LSTM(units,return_sequences = True)
    #     self.lstm_2 = keras.layers.LSTM(units)
    #     """
    #
    #     self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(self.units))
    #
    #     self.dense = keras.layers.Dense(1)
    #
    # def call(self, x, training=None, mask=None):
    #     x = self.embedding(x)
    #     x = self.lstm(x)
    #     x = self.dense(x)
    #
    #     return x
    def __init__(self, units):
        super(RNNModel, self).__init__()
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_review_length)
        self.rnn = keras.layers.Bidirectional(keras.layers.SimpleRNN(units))
        # self.rnn_cell0 = keras.layers.SimpleRNNCell(units, dropout=0.2)
        # self.rnn_cell1 = keras.layers.SimpleRNNCell(units, dropout=0.2)
        self.outlayer = keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        # state0 = self.state0
        # state1 = self.state1
        # for word in tf.unstack(x, axis=1):
        #     out0, state0 = self.rnn_cell0(word, state0, training)
        #     out1, state1 = self.rnn_cell1(out0, state1)
        x = self.rnn(x)
        x = self.outlayer(x)
        return x
    # x = self.outlayer(out1)
    # prob = tf.sigmoid(x)
    # return prob


# model = RNNModel(units, num_classes, num_layers=2)
model = RNNModel(units)
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs, batch_size=batch_size,
          validation_data=(test_data, test_labels))

model.summary()

result = model.evaluate(test_data, test_labels)

# output:loss: 0.6751 - accuracy: 0.8002

# def GRU_Model():
#     model = keras.Sequential([
#         keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_review_length),
#         keras.layers.GRU(32, return_sequences=True),
#         keras.layers.GRU(1, activation='sigmoid', return_sequences=False)
#     ])
#
#     model.compile(optimizer=keras.optimizers.Adam(0.001),
#                   loss=keras.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#
#     return model
#
#
# model = GRU_Model()
# model.summary()
#
# get_ipython().run_cell_magic('time', '',
#                              'history = model.fit(train_data,train_labels,batch_size = batch_size,epochs = epochs,validation_split = 0.1)')
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'validation'], loc='upper left')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
