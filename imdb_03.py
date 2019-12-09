# import pandas as pd
import os
import matplotlib.pyplot  as plt
import tensorflow as tf
# import tensorflow_core
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses
# import keras
# from keras import optimizers, losses, metrics, layers
from keras.datasets import imdb

# tf.__version__
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_words = 30000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_words, output_dim=32, input_length=maxlen),
        # layers.LSTM(32, return_sequences=True),
        # layers.LSTM(1, activation='sigmoid', return_sequences=False)
        layers.GRU(32, return_sequences=True),
        layers.GRU(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=optimizers.Adam(), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


model = lstm_model()
model.summary()
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
result = model.evaluate(x_test, y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()
