import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from pathlib import Path
import pandas as pd

from convnet import ConvNet
from utils import get_data_fast, get_im2col_indices

HIPSTERNET = Path('external/hipsternet')

def test_conv(valsize, seed):
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
    X, Y = make_moons(n_samples=5000, random_state=42, noise=0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

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
    cnn.fit(X, Y, K = 2, minibatch_size = 50, n_iter = 30)

    y_pred = cnn.predict(X_test)
    accs = (y_pred == Y_test).sum() / Y_test.size
    print('Mean accuracy: %f' % accs)

def test_kaggle():
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
    X_test = pd.read_csv('../data/Xte.csv', header = None).as_matrix()[:,:-1]
    cnn = ConvNet()
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"})
    img_1 = X_test[40,:].reshape(3, 32, 32).transpose(1, 2, 0) + [0.25, 0.2, 0.2]
    img_2 = X_test[41,:].reshape(3, 32, 32).transpose(1, 2, 0) + [0.25, 0.2, 0.2]

    # vis_img(X_test[40,:])
    plt.imshow(img_1)
    plt.show()
    plt.imshow(img_2)
    plt.show()

    X_out = cnn.forward_pass(np.array([img_1, img_2]))

    # plt.imshow(X_out[])
    # plt.show()
    # vis_img(X_test[40,:])
    # vis_img(X_out)

def test_cnn():
 
    # vis_img(X_test[40,:])

    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1]
                
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.1)
                    
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)
    # print(X_test.shape)
    
    nb_features = 32 * 32 * 3
    nb_classes = 10

    ch1 = 12#*2
    ch2 = 8#*2
    ch3 = 4#*2
    sizeFC1 = 200

    cnn = ConvNet()
    cnn.add_layer("convlayer", layer_info = {"in_channels": 3,
                                             "out_channels": ch1,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "ReLU"})
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch1,
                                             "out_channels": ch2,
                                             "height": 3,
                                             "width": 3,
                                             "stride": 1,
                                             "padding": 0,
                                            "activation_type": "ReLU"})
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch2,
                                             "out_channels": ch3,
                                             "height": 3,
                                             "width": 3,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": ch3 * 24 * 24, "output_size": sizeFC1, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": sizeFC1, "output_size": nb_classes, "activation_type": "ReLU"})
    # import pudb; pudb.set_trace()  # XXX BREAKPOINT

    cnn.fit(X_train, y_train, K = nb_classes, X_val = X_val, y_val = y_val, minibatch_size = 50, n_iter = 30)
