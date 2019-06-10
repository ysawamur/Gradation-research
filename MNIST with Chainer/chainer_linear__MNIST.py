# モジュールのインポート
import random
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
# MNISTのデータセットをダウンロード
from chainer.datasets import mnist

train, test = mnist.get_mnist(withlabel=True, ndim=1)

# ミニバッチ
batchsize = 100

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

#ネットワーク
class Network(Chain):
    def __init__(self, n_mid_units1=100,n_mid_units2=100, n_out=10):
        super(Network, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units1)
            self.l2 = L.Linear(n_mid_units1, n_mid_units2)
            self.l3 = L.Linear(n_mid_units2, n_out)
            self.dr = 0.5
    def forward(self, x):
        #h1 = F.relu(self.l1(x))
        #h2 = F.relu(self.l2(h1))
        #print(rdm)
        #Dropout
        h1 = F.dropout(F.relu(self.l1(x)), ratio=self.dr)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio=self.dr)
        return self.l3(h2)

model = Network()

gpu_id = -1 # cpuを使うときは-1に設定

if gpu_id >= 0:
    model.to_cpu(gpu_id)

from chainer import optimizers

# 最適化手段の選択
optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, \
                            eps=1e-08, eta=1.0, weight_decay_rate=0, amsgrad=False)
optimizer.setup(model)

from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

max_epoch = 10

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
import time
start = time.time()
while train_iter.epoch < max_epoch:

    # １イテレーションあたりの学習ループ
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch, gpu_id)

    # 予測値の計算
    prediction_train = model(image_train)

    # ソフトマックスエントロピーによる誤差の計算
    loss = F.softmax_cross_entropy(prediction_train, target_train)
    train_losses.append(to_cpu(loss.data))

    #精度の計算
    accuracy = F.accuracy(prediction_train, target_train)
    train_accuracies.append(to_cpu(accuracy.data))

    # 勾配の計算
    model.cleargrads()
    loss.backward()

    # 学習パラメータの更新
    optimizer.update()

    # ----------ここまで---------

    # 予測精度の確認
    if train_iter.is_new_epoch: # 1epochが終わるたびに

        while True:
            test_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch, gpu_id)

            # テストデータの順伝播
            prediction_test = model(image_test)

            # 誤差の計算
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(to_cpu(loss_test.data))

            # 精度の計算
            accuracy = F.accuracy(prediction_test, target_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)
            if test_iter.is_new_epoch:
                #test_iter.reset()
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break
        #認識精度、誤差の表示
        #訓練データ
        train_loss = np.mean(train_losses)
        train_accuracy = np.mean(train_accuracies)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        print('epoch:{:02d} \ntrain_loss:{:.04f} train_accuracy:{:0.4f}' \
                .format(train_iter.epoch, train_loss, train_accuracy))
        #テストデータ
        test_loss = np.mean(test_losses)
        test_accuracy = np.mean(test_accuracies)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)
        print('test_loss:{:.04f} test_accuracy:{:.04f} \n' \
                .format(test_loss, test_accuracy))
elapsed_time = time.time() - start
print(f"実行時間:{elapsed_time}")
#plt.xscale("log")
#plt.yscale("log")
x = np.arange(1, 11)
plt.plot(x, train_acc_list, marker="x", label='train acc')
plt.plot(x, test_acc_list, marker="x", label='test acc')
plt.xlabel("epochs", fontsize=30)
plt.ylabel("accuracy", fontsize=30)
plt.ylim(0.8, 1.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right')
plt.show()

#plt.xscale("log")
#plt.yscale("log")
x = np.arange(1, 11)
plt.plot(x, train_loss_list, marker="x", label='train loss')
plt.plot(x, test_loss_list, marker="x", label='test_loss')
plt.xlabel("epoch", fontsize=30)
plt.ylabel("loss", fontsize=30)
plt.ylim(0, 0.8)
plt.legend(loc='upper right')
plt.show()
