import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Layer:
    # 全连接网络层
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param n_input: 输入节点数
        :param n_neurons: 输出节点数
        :param activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """
        # 通过正态分布初始化网络权值，初始化非常重要，不适合的初始化将导致网络不收敛
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation  # 激活函数类型
        self.last_activation = None  # 激活函数的输出值
        self.error = None  # 用于计算当前层的delta变量的中间变量
        self.delta = None  # 记录当前层的delta变量，用于计算梯度

    def activate(self, x):
        # 向前传播
        r = np.dot(x, self.weights) + self.bias  # X@W+b
        # 通过激活函数得到全连接层的输出
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        # 计算激活函数的输出
        if self.activation is None:
            return r  # 无激活函数
        # ReLU激活函数
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        # tanh
        elif self.activation == 'tanh':
            return np.tanh(r)
        # sigmoid
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        # 计算激活函数的导数
        # 激活函数，导数为1
        if self.activation is None:
            return np.ones_like(r)
        # ReLU函数的导数实现
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        # tanh
        elif self.activation == 'tanh':
            return 1 - r ** 2
        # sigmoid
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r


class NeuralNetwork:
    # 神经网络大类
    def __init__(self):
        self._layers = []  # 网络层对象列表
        self.accuracy = 0

    def add_layer(self, layer):
        # 追加网络层
        self._layers.append(layer)

    def feed_forward(self, X):
        # 向前传播
        for layer in self._layers:
            # 依次通过各个网络层
            X = layer.activate(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        # 反向传播算法实现
        # 向前计算，得到输出值
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):  # 反向循环
            layer = self._layers[i]
            # 如果是输出层
            if layer == self._layers[-1]:  # 对于输出层
                layer.error = y - output  # 计算二分类任务的均方差的导数
                # 计算最后一层的delta，参考输出层的梯度公式
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                # 如果是隐藏层
                next_layer = self._layers[i + 1]  # 得到下一层对象
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                # 关键步骤：计算隐藏层的delta，参考隐藏层的梯度方式
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        # 循环更新权值
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # o_i为上一网络层的输出
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            # 梯度下降算法，delta是公式中的负数，故这里用加号
            layer.weights += layer.delta * o_i.T * learning_rate

    def output(self, y):
        if y[0] >= y[1]:
            return 0
        else:
            return 1

    def train(self, X_train, X_test, y_tain, y_test, learning_rate, max_epochs):
        # 网络训练函数
        # one-hot编码
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        right_samples = 0
        for i in range(max_epochs):  # 训练1000个epoch
            right_samples = 0
            for j in range(len(X_train)):  # 一次训练一个样本
                if self.output(self.feed_forward(X_train[j])) == y_train[j]:
                    right_samples += 1
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            self.accuracy = right_samples / len(X_train)
            if i % 10 == 0:
                # 打印出MSE Loss
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f,Accuracy:%.2f' % (i, float(mse), self.accuracy))

                # 统计并打印准确率
                # print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test), y_test.flatten()) * 100))

        return mses

    def predict(self, X_test, y_test):
        right_samples = 0
        result = []
        for i in range(len(X_test)):
            if self.output(self.feed_forward(X_test[i])) == y_test[i]:
                right_samples += 1
                result.append(1)
            else:
                result.append(0)
        print('Accuracy:%.2f' % (right_samples / len(X_test)))
        return result
