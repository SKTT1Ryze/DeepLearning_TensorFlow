import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batchsz = 128
h_dim = 20
lr = 1e-4
epochs = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.shuffle(batchsz * 5).batch(batchsz)


class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    pass

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat

    pass


def main():
    model = AE()
    model.build(input_shape=(4, 784))
    model.summary()
    optimizer = tf.optimizers.Adam(lr=lr)
    for epoch in range(epochs):
        for step, x in enumerate(train_db):
            x = tf.reshape(x, [-1, 784])
            with tf.GradientTape() as tape:
                x_rec_logits = model(x)
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
                rec_loss = tf.reduce_mean(rec_loss)
            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(rec_loss))
                pass
            x = next(iter(test_db))
            logits = model(tf.reshape(x, [-1, 784]))
            x_hat = tf.sigmoid(logits)
            x_hat = tf.reshape(x_hat, [-1, 28, 28])
            x_concat = tf.concat([x, x_hat], axis=0)
            x_concat = x_concat.numpy() * 255.
            x_concat = x_concat.astype(np.uint8)
            save_images(x_concat, 'ae_image/rec_epoch_%d.png' % epoch)

            pass


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
            pass
        pass
    new_im.save(name)


if __name__ == '__main__':
    main()
