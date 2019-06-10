import sys
sys.path.append("./deep learning/")
import numpy as np
from dataset.mnist import load_mnist
from Three_layer import ThreelayerNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

network = ThreelayerNet(input_size=784, hidden_size=100, output_size=10)

iters_num = 6001
train_size = 60000
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
import time
start = time.time()
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
elapsed_time = time.time() - start
print(f"実行時間:{elapsed_time:04f}")

#plt.yscale("log")
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker="o", label='train acc')
plt.plot(x, test_acc_list, marker="s", label='test acc')
plt.xlabel("epochs", fontsize=30)
plt.ylabel("accuracy", fontsize=30)
plt.ylim(0.0, 1.0)
#plt.tick_params(labelsize = 20)
plt.legend(loc='lower right', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker="o", label='train acc')
plt.plot(x, test_acc_list, marker="s", label='test acc')
plt.xlabel("epochs", fontsize=30)
plt.ylabel("accuracy", fontsize=30)
plt.ylim(0.8, 1.0)
#plt.tick_params(labelsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.show()
