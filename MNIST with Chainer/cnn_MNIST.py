import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


train, test = chainer.datasets.get_mnist(ndim=3)
x_train = train._datasets[0]
t_train = train._datasets[1]
x_test = test._datasets[0]
t_test = test._datasets[1]


class CNN(Chain):
    # Constructor
    def __init__(self, initializer = None):
        super().__init__(
            Convol1 = L.Convolution2D(1,30,ksize=5, stride=1, pad=0),
            Convol2 = L.Convolution2D(30,30,ksize=5, stride=1, pad=0),
            Affine2 = L.Linear(480, 100, initialW = initializer),
            Affine3 = L.Linear(100, 10, initialW = initializer)
        )

    # Forward operation
    def __call__(self, x, t = None):
        c1 = self.Convol1(x)
        z1 = F.relu(c1)
        m1 = F.max_pooling_2d(z1,2)
        c2 = self.Convol2(m1)
        z2 = F.relu(c2)
        m2 = F.max_pooling_2d(z2,2)
        #print(c1.shape,z1.shape,m1.shape)
        z3 = F.dropout(F.relu(self.Affine2(m2)),ratio=0.1) # Affine2 - Sigmoid2
        a3 = self.Affine3(F.dropout(z3,ratio=0.1)) # Affine3
        if chainer.config.train:
            return F.softmax_cross_entropy(a3, t) # Softmax3 with cross entropy error
        else:
            return F.softmax(a3) # Softmax3
        return a3

model = CNN(initializer = chainer.initializers.HeNormal())
optimizer = chainer.optimizers.Adam()
optimizer.use_cleargrads()
optimizer.setup(model)

# Set parameters and initialiation
iters_num = 6001
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)
# Training and evaluation
train_acc_list = []
test_acc_list = []
for i in range(iters_num):
    # Set mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
#    x_batch = chainer.cuda.to_gpu(x_train[batch_mask], device = gpu_device)
#    t_batch = chainer.cuda.to_gpu(t_train[batch_mask], device = gpu_device)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Forward operation
    loss = model(x_batch, t_batch)

    # Backward operation
    model.cleargrads()
    loss.backward()

    # Update parameters
    optimizer.update()

    # Evaluation
    if i % iter_per_epoch == 0 or i == iters_num - 1:
        # Turn training flag off
        chainer.config.train = False

        # Evaluate training set
        y_train = []
        for s in range(0, train_size, batch_size):
#            x_batch = chainer.cuda.to_gpu(x_train[s:s + batch_size])
#            y_train.extend(chainer.cuda.to_cpu(model(x_batch).data).tolist())
            x_batch = x_train[s:s + batch_size]
            y_train.extend(model(x_batch).data.tolist())

        # Evaluate test set
        y_test = []
        for s in range(0, test_size, batch_size):
#            x_batch = chainer.cuda.to_gpu(x_test[s:s + batch_size])
#            y_test.extend(chainer.cuda.to_cpu(model(x_batch).data).tolist())
            x_batch = x_test[s:s + batch_size]
            y_test.extend(model(x_batch).data.tolist())

        # Compute accuracy
        train_acc = F.accuracy(np.array(y_train), t_train).data
        test_acc = F.accuracy(np.array(y_test), t_test).data
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i/600, train_acc, test_acc)

        # Turn training flag on
        chainer.config.train = True
#from chainer import serializers

#serializers.save_npz('my_mnist.model', model)

#plt.yscale("log")
x = np.arange(0, 11)
plt.plot(x, train_acc_list, marker="x", label='train acc')
plt.plot(x, test_acc_list, marker="x", label='test acc')
plt.xlabel("epochs", fontsize=30)
plt.ylabel("accuracy", fontsize=30)
plt.ylim(0.95, 1.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.show()

#plt.yscale("log")
x = np.arange(0, 11)
plt.plot(x, train_acc_list, marker="x", label='train acc')
plt.plot(x, test_acc_list, marker="x", label='test acc')
plt.xlabel("epochs", fontsize=30)
plt.ylabel("accuracy", fontsize=30)
plt.ylim(0.97, 1.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.show()

x = np.arange(0, 11)
plt.plot(x, train_acc_list, marker="x", label='train acc')
plt.plot(x, test_acc_list, marker="x", label='test acc')
plt.xlabel("epochs", fontsize=30)
plt.ylabel("accuracy", fontsize=30)
plt.ylim(0.98, 1.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.show()
