# import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import hipsternet.input_data as input_data
from pathlib import Path
import pandas as pd
from datetime import datetime

from convnet import ConvNet
from utils import get_data_fast, get_im2col_indices, prepro_mnist, prepro_cifar, data_augmentation
import pickle

HIPSTERNET = Path('external/hipsternet')
DUMPFOLDER = Path('dumps')

def test_conv_layer(valsize, seed):
    # Data loading
    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1]
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = valsize)
                    
    ch1 = 12
    cnn = ConvNet()
    cnn.add_layer("convlayer", layer_info = {"in_channels": 3,
                                             "out_channels": ch1,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "None"})

    np.random.seed(100)
    xtr_res = X_train.reshape(-1, 3, 32,32)
    input_batch = xtr_res[:3, :, :, :]
    V = cnn.layers[0].forwardprop(input_batch)
    # Traspose img to [batch_size, channels, im_height, im_width]
    # W1 = np.random.normal(0, 1, (12, 3, 5, 5)) # random weights, 12 filters
    
    W1 = np.ones((12,3,5,5)) * 0.01
    b1 = np.ones([12]) * 0.01
    # Doing forward pass
    er1 = np.random.random(V.shape)
    B = cnn.layers[0].backprop(V-er1, V, input_batch)

    import sys
    sys.path.append(str(HIPSTERNET))
    import hipsternet.layer as hl
    out, cache =  hl.conv_forward(input_batch, W1, np.expand_dims(b1, 1), stride = 1, padding = 0)
    OUT = hl.conv_backward(out - er1, cache)

    print('hip_to_mine_conv_mistake = {}'.format(np.mean(V - out)))
    print('hip_to_mine_grad_mistake = {}'.format(np.mean(B[0] - OUT[1])))
    print('hip_to_mine_grad_sum  = {} {}'.format(np.sum(B[0]), np.sum(OUT[1])))

    # sys.exit()

    # TENSORFLOW RELATED STUFF
    # INPUTS: batch|channels|height|width --> batch|height|width|channels
    tf_input_batch = input_batch.transpose(0, 2, 3, 1)
    # WEIGHTS: out_channels|in_channels|height|width ---> heigh|width|in_channels|out_channels
    tf_W1 = W1.transpose(2, 3, 1, 0)
    # LOSS: like inputs
    tf_er1 = er1.transpose(0, 2, 3, 1)

    # Convolution
    ibatch_ = tf.placeholder(tf.float32, shape=tf_input_batch.shape)
    W_ = tf.Variable(initial_value=tf_W1, dtype=tf.float32)
    b_ = tf.Variable(initial_value=b1, dtype=tf.float32)
    conv  = tf.nn.conv2d(ibatch_, W_, strides=[1,1,1,1], padding='VALID', use_cudnn_on_gpu=False)
    pre_activation = tf.nn.bias_add(conv, b_)
    # conv2 = tf.nn.relu(pre_activation)
    conv2 = pre_activation

    # Conv gradients
    er1_ = tf.placeholder(tf.float32, shape=tf_er1.shape)
    silly_loss = conv2-er1_
    tfgrad = tf.gradients(silly_loss, W_)

    # more grads
    opt = tf.train.GradientDescentOptimizer(0.001)
    grads = opt.compute_gradients(silly_loss)

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            sess.run(tf.global_variables_initializer())
            V_, B_, loss_, grads_ = sess.run([conv2, tfgrad, silly_loss, grads], feed_dict={
                ibatch_:tf_input_batch.astype('float32'),
                er1_: tf_er1.astype('float32')})

    print('TF_conv_mistake = {}'.format(np.mean(V.transpose(0, 2, 3, 1) - V_)))
    print('TF_grad_mistake = {}'.format(np.mean(B[0].transpose(2, 3, 1, 0) - B_)))
    print('TF_grad_sum  = {} {}'.format(np.sum(B[0].transpose(2, 3, 1, 0)), np.sum(B_)))

    # plt.imshow(V[:, :, 3])

def test_moons():
    X, Y = make_moons(n_samples=7000, random_state=42, noise=0.1)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, random_state=42, test_size = 0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, random_state=42, test_size = 0.1)

    # run_nn(X_train, X_test, Y_train, Y_test)

    np.random.seed(228)
    size1 = 2
    size2 = 100
    size3 = 40
    size4 = 2

    cnn = ConvNet()
    cnn.add_layer("fclayer", layer_info = {"input_size": size1, "output_size": size2, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size2, "output_size": size3, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size3, "output_size": size4, "activation_type": "None"})
    cnn.fit(X, Y, X_cv = X_val, y_cv = Y_val, K = 2, minibatch_size = 50, n_iter = 100, print_every_proc = 100, step_size=0.01, use_vanila_sgd=True)

    # print(Y_train)
    # print(Y_test)

    y_pred = cnn.predict(X_test)
    accs = (y_pred == Y_test).sum() / Y_test.size
    print('Mean accuracy: %f' % accs)


def test_kaggle_fcnn():
    X_train_full = pd.read_csv('../data/Xtr.csv', header = None).as_matrix()[:,:-1]
    y_train_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
    X_test = pd.read_csv('../data/Xte.csv', header = None).as_matrix()[:,:-1]

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    # vis_img(X_test[40,:])

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)

    nb_features = 32 * 32 * 3
    nb_classes = 10
    size1 = nb_features
    size2 = 2000
    size3 = 2000
    size4 = 500
    size5 = 200
    size6 = nb_classes

    cnn = ConvNet()
    cnn.add_layer("fclayer", layer_info = {"input_size": size1, "output_size": size2, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size2, "output_size": size3, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size3, "output_size": size4, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size4, "output_size": size5, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size5, "output_size": size6, "activation_type": "None"})
    # cnn.add_layer("fclayer", layer_info = {"input_size": size6, "output_size": size7, "activation_type": "None"})

    #cnn.fit(X_train, y_train, K = nb_classes, X_val = X_val, y_val = y_val, minibatch_size = 50, n_iter = 30)
    #y_test = cnn.predict(X_test)
    #y_test.dump("Yte.dat")

def test_poollayer():
    X_test = pd.read_csv('Data/Xte.csv', header = None).as_matrix()[:,:-1]
    cnn = ConvNet()
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"})

    cnn.layers[0].assert_pool_layer()

def test_kaggle_cnn():
    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.8)

    nb_samples, data_length, nb_classes = X_train.shape[0], X_train.shape[1], y_train.max() + 1
    img_shape = (3, 32, 32)

    X_train, X_val, X_test = prepro_cifar(X_train, X_val, X_test, img_shape)

    # X_train, y_train = data_augmentation(X_train, y_train)

    print("X_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("X_val.shape = ", X_val.shape)
    print("y_val.shape = ", y_val.shape)
    print("X_test.shape = ", X_test.shape)

    dump_folder = DUMPFOLDER/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dump_folder.mkdir()


    ch1 = 32
    ch2 = 64
    ch3 = 256
    nb_classes = 10

    cnn = ConvNet()
    cnn.set_img_shape(img_shape)
    cnn.add_layer("convlayer", layer_info = {"in_channels": img_shape[0],
                                             "out_channels": ch1,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 2,
                                             "activation_type": "ReLU"}) # 32 x 32 x ch1
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"}) # 16 x 16 x ch1
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch1,
                                             "out_channels": ch2,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 2,
                                            "activation_type": "ReLU"}) # 16 x 16 x ch2
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"}) # 8 x 8 x ch2
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch2,
                                             "out_channels": ch3,
                                             "height": 8,
                                             "width": 8,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "ReLU"}) # 1 x 1 x ch3
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch3,
                                             "out_channels": nb_classes,
                                             "height": 1,
                                             "width": 1,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "None"}) # 1 x 1 x 10

    cnn.fit(X_train,
            y_train,
            K = nb_classes,
            X_cv = X_val,
            y_cv = y_val,
            minibatch_size = 50,
            n_iter = 30,
            step_size = 0.1,
            use_vanila_sgd = True,
            print_every_proc = 1,
            path_for_dump = dump_folder)

    y_test = cnn.predict(X_test)
    pickle.dump(y_test, (dump_folder/"Yte.dat").open('wb'))


def test_mnist():
    # get kaggle data
    # X_train_full = get_data_fast("Xtr")[:,:-1]
    # X_test = get_data_fast("Xte")[:,:-1]
    # y_train_full = get_data_fast("Ytr")[:,1].astype(int)

    # get CIFAR_orig data
    # data = unpickle("../cifar_orig/data_batch_5")
    # X_train_full = data[b'data']
    # y_train_full = np.array(data[b'labels'])

    # X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    # get MNIST data
    mnist = input_data.read_data_sets('Data/MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    sampled_indexes_train = np.random.choice(X_train.shape[0], 5000)
    X_train = X_train[sampled_indexes_train,:]
    y_train = y_train[sampled_indexes_train]

    X_val, y_val = mnist.validation.images, mnist.validation.labels
    sampled_indexes_val = np.random.choice(X_val.shape[0], 1000)
    X_val = X_val[sampled_indexes_val,:]
    y_val = y_val[sampled_indexes_val]

    X_test, y_test = mnist.test.images, mnist.test.labels

    nb_samples, data_length, nb_classes = X_train.shape[0], X_train.shape[1], y_train.max() + 1
    img_shape = (1, 28, 28)

    X_train, X_val, X_test = prepro_mnist(X_train, X_val, X_test)

    X_train = X_train.reshape(-1, *img_shape)
    X_val = X_val.reshape(-1, *img_shape)
    X_test = X_test.reshape(-1, *img_shape)

    print("X_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("X_val.shape = ", X_val.shape)
    print("y_val.shape = ", y_val.shape)
    print("X_test.shape = ", X_test.shape)


    ch1 = 32
    ch2 = 64
    ch3 = 128
    nb_classes = 10

    cnn = ConvNet()
    cnn.set_img_shape(img_shape)
    cnn.add_layer("convlayer", layer_info = {"in_channels": img_shape[0],
                                             "out_channels": ch1,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 2,
                                             "activation_type": "ReLU"}) # 28 x 28 x ch1
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"}) # 14 x 14 x ch1
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch1,
                                             "out_channels": ch2,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 2,
                                            "activation_type": "ReLU"}) # 14 x 14 x ch2
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"}) # 7 x 7 x ch2
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch2,
                                             "out_channels": ch3,
                                             "height": 7,
                                             "width": 7,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "ReLU"}) # 1 x 1 x ch3
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch3,
                                             "out_channels": nb_classes,
                                             "height": 1,
                                             "width": 1,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "None"}) # 1 x 1 x 10
    cnn.fit(X_train, y_train, K = nb_classes, X_cv = X_val, y_cv = y_val, minibatch_size = 50, n_iter = 30, step_size=0.01, use_vanila_sgd=True)
