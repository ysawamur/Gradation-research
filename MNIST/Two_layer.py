import sys
sys.path.append("./deep learning/")
import numpy as np
from layers import *
from gradient import *
from collections import OrderedDict

class TwolayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmod'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastlayer = Softmaxwithloss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def gradient1(self, x, t):
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self, layers.values)
        layers.reverse()
        for layer in layers:
            dout =layer.backward(dout)

        grads = {}
        grads['b1'] = self.layers['Affine1'].db
        grads['W1'] = self.layers['Affine1'].dw
        grads['b2'] = self.layers['Affine2'].db
        grads['W2'] = self.layers['Affine2'].dw

        return grads
