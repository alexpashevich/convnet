from utils import get_data_fast
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.model_selection import train_test_split

def test_tensorflow_full():

    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1]

    X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    xtr_res = X_train.reshape(-1, 3, 32,32)

    input_batch = xtr_res[:50, :, :, :]

    # er1 = np.random.random(V.shape)

    # TENSORFLOW RELATED STUFF
    # INPUTS: batch|channels|height|width --> batch|height|width|channels
    tf_input_batch = input_batch.transpose(0, 2, 3, 1)
    # WEIGHTS: out_channels|in_channels|height|width ---> heigh|width|in_channels|out_channels
    # tf_W1 = W1.transpose(2, 3, 1, 0)
    # LOSS: like inputs
    # tf_er1 = er1.transpose(0, 2, 3, 1)

    # Convolution
    ibatch_ = tf.placeholder(tf.float32, shape=tf_input_batch.shape)

    # CONV_Layer 1
    W1_h = 5
    W1_w = 5
    n_filter_1 = 5 # out_channels
    n_chann_1 = 3   # in channels
    W1 = np.random.randn((W1_h, W1_w, n_chann_1, n_filter_1)) * 0.01
    b1 = np.random.randn(n_filter_1) * 0.01

    with tf.variable_scope('conv1') as scope:
        W = tf.Variable(initial_value=W1, dtype=tf.float32)
        b = tf.Variable(initial_value=b1, dtype=tf.float32)
        conv  = tf.nn.conv2d(ibatch_, W, strides=[1,1,1,1], padding='VALID', use_cudnn_on_gpu=False)
        pre_activation = tf.nn.bias_add(conv, b)
        conv1 = tf.nn.relu(pre_activation)

    # Conv Layer 2
    W2_h = 3
    W2_w = 3
    n_filter_2 = 8
    n_chann2 = n_filter_1
    W2 = np.random.randn((W2_h, W2_w, n_chann2, n_filter_2)) * 0.01
    b2 = np.random.randn((n_filter_2)) * 0.01

    with tf.variable_scope('conv2') as scope:
        W = tf.Variable(initial_value = W2, dtype = tf.float32)
        b = tf.Variable(initial_value = b2, dtype = tf.float32)
        conv = tf.nn.conv2d(conv1, W, strides = [1,1,1,1], padding = 'VALID', use_cudnn_on_gpu = False)
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, b))

    # CONV Layer 3

    W3_h = 3
    W3_w = 3
    n_filter_3 = 4
    n_chann3 = n_filter_2
    W3 = np.random.randn((W3_h, W3_w, n_chann3, n_filter_3)) * 0.01
    b3 = np.random.randn((n_filter_3)) * 0.01

    with tf.variable_scope('conv2') as scope:
        W = tf.Variable(initial_value = W3, dtype = tf.float32)
        b = tf.Variable(initial_value = b3, dtype = tf.float32)
        conv = tf.nn.conv2d(conv2, W, strides = [1,1,1,1], padding = 'VALID', use_cudnn_on_gpu = False)
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, b))

    # Fclayer1

    dim_in1 = n_filter_3*24*24
    dim_out1 = 200 
    W4 = np.random.randn((dim_in1, dim_out1)) * 0.01 
    b4 = np.random.randn(dim_out1) * 0.01

    with tf.variable_scope('fc1') as scope:
        W = tf.Variable(initial_value = W4, dtype = tf.float32)
        b = tf.Variable(initial_value = b4, dtype = tf.float32)
        fc1 = tf.nn.relu( tf.add(tf.matmul(conv3, W), b, name = scope.name))

    # Fclayer2 

    dim_in2 = dim_out1
    dim_out2 = 10
    W5 = np.random.randn(dim_in2, dim_out2) * 0.01 
    b5 = np.random.randn([dim_out2]) * 0.01

    with tf.variable_scope('fc2') as scope:
        W = tf.Variable(initial_value = W5, dtype = tf.float32)
        b = tf.Variable(initial_value = b5, dtype = tf.float32)
        fc2 = tf.add(tf.matmul(fc1, W), b, name = scope.name)

        import pudb; pudb.set_trace()  # XXX BREAKPOINT


    # Now compute prediction 

    # Conv gradients
    # er1_ = tf.placeholder(tf.float32, shape=tf_er1.shape)

    def CE_loss(y_true, y_pred):
        # cross entropy loss (multinomial regression)
        y_pred[y_pred < 1e-7] = 1e-7 # we clip values in order to prevent log of very small value
        return -np.sum(y_true * np.log(y_pred))

    def CE_loss_batch(y_true_batch, y_pred_batch):
        loss = 0
        # y_pred_batch[y_pred_batch < 1e-7] = 1e-7 # we clip values in order to prevent log of very small value
        for y_true, y_pred in zip(y_true_batch, y_pred_batch):
            loss += CE_loss(y_true, y_pred)
        return loss

    #silly_loss = CE_loss(conv2 - )
    tfgrad = tf.gradients(silly_loss, W_)

    # more grads
    opt = tf.train.GradientDescentOptimizer(0.001)
    grads = opt.compute_gradients(silly_loss)

    # with tf.Session() as sess:
    #         # with tf.device('/cpu:0'):
    #      with tf.device('/GPU:0'):
    #           sess.run(tf.global_variables_initializer())
    #           V_, B_, loss_, grads_ = sess.run([conv2, tfgrad, silly_loss, grads], feed_dict={
    #              ibatch_:tf_input_batch.astype('float32'),
    #               er1_: tf_er1.astype('float32')})

    # with tf.Session() as sess:
    #     with tf.device('/cpu:0'):
    #         sess.run(tf.global_variables_initializer())
    #         V_, B_, loss_, grads_ = sess.run([conv2, tfgrad, silly_loss, grads], feed_dict = {
    #             ibatch_:tf_input_batch.astype('float32')})

def slim_training(X_train, y_train, X_val, y_val, sess):
    xtr_res = X_train.reshape(-1, 3, 32,32)

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    input_batch = xtr_res[:50, :, :, :]
    tf_input_batch = input_batch.transpose(0, 2, 3, 1)
    sess.run([train_op], feed_dict = {
            input :tf_input_batch.astype('float32')})

def test_tensorflow_slim():

    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1]

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    learning_rate = 0.001

    # Convolution
    input = tf.placeholder(tf.float32, shape=(None, 3, 32, 32))
    labels = tf.placeholder(tf.float32, shape=(None, 10))

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(0.0, 0.001),
            weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(input, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.fully_connected(net, 1024, scope='fc3')
        predictions = slim.fully_connected(net, 10, scope='fc4')

    loss = slim.losses.softmax_cross_entropy(predictions, labels)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        slim_training(X_train, y_train, X_val, y_val, sess)
