import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets  import tuple_dataset
from chainer.backends.cuda import to_cpu
from chainer.backends.cuda import to_gpu
import cupy as cp
from chainer import serializers

FILTER = 50
class CNN(Chain):
    # Constructor
    def __init__(self, initializer = None):
        super().__init__(
            Convol1 = L.Convolution2D(None,FILTER,ksize=5, stride=1, pad=0, initialW = initializer),
            Convol2 = L.Convolution2D(FILTER,FILTER,ksize=5, stride=1, pad=0, initialW = initializer),
            Convol3 = L.Convolution2D(FILTER,FILTER,ksize=6, stride=1, pad=0, initialW = initializer),
            Convol4 = L.Convolution2D(FILTER,FILTER,ksize=5, stride=1, pad=0, initialW = initializer),
            Convol5 = L.Convolution2D(FILTER,FILTER,ksize=5, stride=1, pad=0, initialW = initializer),
            Convol6 = L.Convolution2D(FILTER,FILTER,ksize=5, stride=1, pad=0, initialW = initializer),
            Affine2 = L.Linear(None, 100, initialW = initializer),
            Affine3 = L.Linear(100, 1, initialW = initializer)
        )

    # Forward operation
    def __call__(self, x, t = None):
        c1 = self.Convol1(x)
        z1 = F.relu(c1)
#        m1 = F.max_pooling_2d(z1,2)
        m1 = F.average_pooling_2d(z1,2)
        c2 = self.Convol2(m1)
        z2 = F.relu(c2)
#        m2 = F.max_pooling_2d(z2,2)
        m2 = F.average_pooling_2d(z2,2)
        c3 = self.Convol3(m2)
        z3 = F.relu(c3)
#        m3 = F.max_pooling_2d(z3,2)
        m3 = F.average_pooling_2d(z3,2)
        c4 = self.Convol4(m3)
        z4 = F.relu(c4)
#        m4 = F.max_pooling_2d(z4,2)
        m4 = F.average_pooling_2d(z4,2)
        c5 = self.Convol5(m4)
        z5 = F.relu(c5)

        m5 = F.average_pooling_2d(z5,2)
        # c6 = self.Convol6(m5)
        # z6 = F.relu(c6)
        # m6 = F.average_pooling_2d(z6, 2)

        z7 = F.relu(self.Affine2(m5))


        a5 = self.Affine3(z7)
#        z5 = F.dropout(F.relu(self.Affine2(m4)),ratio=0.5) # Affine2 - Sigmoid2
#        a5 = self.Affine3(F.dropout(z5,ratio=0.5))


        if chainer.config.train:
            return F.mean_squared_error(a5.reshape(len(a5),), t) # mean_squared_error
        else:
            return a5.reshape(len(a5),)# identify

if __name__ == '__main__':
    SHAPE = 384
    FILTER = 50
    if SHAPE == 128:
        s_img = np.load('/work/ysawamura/solar_image/x_data_r.npy')
    else:
        s_img = np.load('/work/ysawamura/solar_image/x_data_r'+str(SHAPE)+'.npy')
    flux =  np.load('/work/ysawamura/solar_image/t_data.npy')
    s_img = s_img.reshape(len(s_img), 1, SHAPE, SHAPE)
    s_img = s_img.astype(np.float32)
    flux = flux / 10 **22
    flux = flux.astype(np.float32)
    update_count = 0

    mask = np.arange(5914)
    np.random.seed(100)
    mask = np.random.permutation(mask)
    s_img = s_img[:5914]
    s_img_varidation = s_img[5914:]
    flux = flux[:5914]
    flux_varidation = flux[5914:]
    mask1 = mask[:5000]
    mask2 = mask[5000:]
    s_img_train = s_img[mask1]
    s_img_test = s_img[mask2]
    flux_train = flux[mask1]
    flux_test = flux[mask2]
    train = tuple_dataset.TupleDataset(s_img_train, flux_train)
    test = tuple_dataset.TupleDataset(s_img_test, flux_test)
    x_train = train._datasets[0]
    t_train = train._datasets[1]
    x_test = test._datasets[0]
    t_test = test._datasets[1]


    model_load = False

    if model_load:
        model = serializers.load_npz('my_mnist.model', CNN())
    else:
        model = CNN(initializer = chainer.initializers.HeNormal())

    gpu_id = 0# cpuを使うときは-1に設定

    if gpu_id >= 0:
        model.to_gpu(gpu_id)


    optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, \
                                eps=1e-08, eta=1.0, weight_decay_rate=0, amsgrad=False)

    optimizer.use_cleargrads()
    optimizer.setup(model)

    iters_num = 10001
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    batch_size = 50
    per_epoch = train_size / batch_size
    iter_per_epoch = max(per_epoch, 1)

    # Training and evaluation
    train_losses = []
    test_losses = []
    train_r2 = []
    test_r2 = []

    import time
    start = time.time()
    for i in range(iters_num):
        chainer.config.train = True
        start2 = time.time()

        # gpu を使う場合
        if gpu_id>=0:
            batch_mask = np.random.choice(train_size, batch_size, False)
            x_batch = chainer.cuda.to_gpu(x_train[batch_mask], device = gpu_id)
            t_batch = chainer.cuda.to_gpu(t_train[batch_mask], device = gpu_id)

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
                    x_batch = chainer.cuda.to_gpu(x_train[s:s + batch_size])
                    y_train.extend(chainer.cuda.to_cpu(model(x_batch).data).tolist())

                # Evaluate test set
                y_test = []
                for s in range(0, test_size, batch_size):
                    x_batch = chainer.cuda.to_gpu(x_test[s:s + batch_size])
                    y_test.extend(chainer.cuda.to_cpu(model(x_batch).data).tolist())


                train_loss = F.mean_squared_error(cp.asarray(y_train, dtype=np.float32), cp.asarray(t_train,dtype=np.float32))
                train_R2 = F.r2_score(cp.asarray(y_train, dtype=np.float32),cp.asarray(t_train,dtype=np.float32))
                train_losses.append(to_cpu(train_loss.data))
                train_r2.append(to_cpu(train_R2.data))
                train_loss = cp.asnumpy(train_loss.data)
                train_R2 = cp.asnumpy(train_R2.data)
                test_loss = F.mean_squared_error(cp.asarray(y_test, dtype=np.float32), cp.asarray(t_test, dtype=np.float32))
                test_R2 = F.r2_score(cp.asarray(y_test, dtype=np.float32),cp.asarray(t_test,dtype=np.float32))
                test_losses.append(to_cpu(test_loss.data))
                test_r2.append(to_cpu(test_R2.data))
                test_loss = cp.asnumpy(test_loss.data)
                test_R2 = cp.asnumpy(test_R2.data)
                elapsed_time2 = time.time() - start2
                epoch = i / per_epoch
                epoch_predict = y_train + y_test
                print('epoch:{:.0f} \ntrain_loss:{:.04f} test_loss:{:0.4f} time:{:02f} '.format(epoch, train_loss, test_loss, elapsed_time2))
                print('train_R2:{:.04f} test_R2:{:0.4f} '.format(train_R2, test_R2))
                if epoch > 30:
                    if test_r2[int(epoch)] == min(test_r2):
                        serializers.save_npz('/work/ysawamura/solar_image/solar_cnn.model'+ str(SHAPE), model)
                        print(update_count)
                        update_count += 1


                chainer.config.train = True

        # cpu を使う場合
        else:
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
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
                    x_batch = x_train[s:s + batch_size]
                    y_train.extend(model(x_batch).data.tolist())

                # Evaluate test set
                y_test = []
                for s in range(0, test_size, batch_size):
                    x_batch = x_test[s:s + batch_size]
                    y_test.extend(model(x_batch).data.tolist())


                train_R2 = F.r2_score(np.array(y_train,t_train))
                train_loss = F.mean_squared_error(np.array(y_train), t_train)
                train_losses.append(to_cpu(train_loss.data))
                train_r2.append(to_cpu(train_R2.data))
                train_loss = cp.asnumpy(train_loss.data)
                train_R2 = cp.asnumpy(train_R2.data)
                test_loss = F.mean_squared_error(np.array(y_test), t_test)
                test_R2 = F.r2_score(np.array(y_test,t_test))
                test_losses.append(to_cpu(test_loss.data))
                test_r2.append(to_cpu(test_R2.data))
                test_loss = test_loss.data
                test_R2 = test_R2.data
                elapsed_time2 = time.time() - start2
                epoch = i / per_epoch
                epoch_predict = y_train + y_test
                print('epoch:{:.0f} \ntrain_loss:{:.04f} test_loss:{:0.4f} time:{:02f} '.format(epoch, train_loss, test_loss, elapsed_time2))
                print('train_R2:{:.04f} test_R2:{:0.4f} '.format(train_R2, test_R2))
                if epoch > 30:
                    if test_r2[int(epoch)] == min(test_r2):
                        serializers.save_npz('/work/ysawamura/solar_image/solar_cnn.model'+ str(SHAPE), model)
                        print(update_count)
                        update_count += 1


                chainer.config.train = True

    elapsed_time = time.time() - start
    print(f"time:{elapsed_time}")
    print('train_loss:{:.04f} test_loss:{:0.4f}'.format(min(train_losses), min(test_losses)))
    print('train_r2:{:.04f} test_r2:{:0.4f}'.format(max(train_r2), max(test_r2)))
