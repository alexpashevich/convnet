import numpy as np
import pandas as pd
import math
import sys
# sys.path.append('/media/d/study/Grenoble/courses/advanced_learning_models/Competition/temp/hipsternet')
sys.path.append("../hipsternet")
import hipsternet.layer as hl
import hipsternet.input_data as input_data
from sklearn.utils import shuffle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import matplotlib.pyplot as plt
from timeit import default_timer as timer

from utils import vis_img, get_data_fast, get_im2col_indices
from fclayer import FCLayer
from poollayer import PoolLayer
from convlayer import ConvLayer


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

def square_loss(y_true, y_pred): # not used now
    # square loss
    return 1. / 2 * np.sum((y_true - y_pred) ** 2)


def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T


class ConvNet:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, layer_type, layer_info):
        ''' add a layer to the NN '''
        if layer_type == "fclayer":
            self.layers.append(FCLayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "convlayer":
            self.layers.append(ConvLayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "poollayer":
            self.layers.append(PoolLayer(layer_info))
            self.nb_layers += 1
        else:
            print("error: unknown layer type")

    def forward_pass_hipster(self, X):
        cur_input_hipster = X
        outputs_hipster = []
        cache_hipster = []
        outputs_hipster.append(cur_input_hipster)

        for layer in self.layers:
            # print("forward_pass iter")
            if type(layer) is ConvLayer and len(cur_input_hipster.shape) < 4:
                # X -> ConvLayer, we have to resize the input
                cur_input_hipster = cur_input_hipster.reshape(-1, 3, 32, 32)

            cur_input_hipster, cache = hl.conv_forward(cur_input_hipster, layer.W, np.expand_dims(layer.b, 1), stride = layer.stride, 
                                                        padding = layer.padding)
            outputs_hipster.append(np.array(cur_input_hipster))
            cache_hipster.append(cache)

        if len(cur_input_hipster.shape) > 2:
            # we used a NN without FC layers
            cur_input_hipster = cur_input_hipster.reshape(-1, cur_input_hipster.shape[1] * cur_input_hipster.shape[2] * cur_input_hipster.shape[3])

        # the softmax layer, we subtract maximum to avoid overflow
        cur_input_hipster = softmax(cur_input_hipster)
        outputs_hipster.append(cur_input_hipster)

        return outputs_hipster, cache_hipster

    def forward_pass(self, X):
        ''' return the input data X and outputs of every layer '''
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        
        for layer in self.layers:
            # print("forward_pass iter")
            if type(layer) is ConvLayer and len(cur_input.shape) < 4:
                # X -> ConvLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, 3, 32, 32)

            if type(layer) is FCLayer and len(cur_input.shape) > 2:
                # ConvLayer -> FCLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, cur_input.shape[1] * cur_input.shape[2] * cur_input.shape[3])

            cur_input = layer.forwardprop(cur_input)
            outputs.append(cur_input)

        if len(cur_input.shape) > 2:
            # we used a NN without FC layers
            cur_input = cur_input.reshape(-1, cur_input.shape[1] * cur_input.shape[2] * cur_input.shape[3])

        # the softmax layer, we subtract maximum to avoid overflow
        cur_input = softmax(cur_input)
        # cur_input = np.exp(cur_input - np.max(cur_input)) / np.outer(np.exp(cur_input - np.max(cur_input)).sum(axis=1), np.ones(cur_input.shape[1]))
        outputs.append(cur_input)

        return outputs

    def backward_pass_hipster(self, errors_batch_hipster, outputs_batch_hipster, cache_hipster):
        ''' do the backward pass and return grads for W update '''
        i = 1

        grad_W_hipster = len(self.layers) * [None]
        grad_b_hipster = len(self.layers) * [None]

        for layer in reversed(self.layers):

            if type(layer) is ConvLayer:
                type_name = "convlayer"
            else:
                type_name = "fclayer"

            cur_out_hipster = np.array(outputs_batch_hipster[-i])
            prev_out_hipster = np.array(outputs_batch_hipster[-1 - i])

            if type(layer) is ConvLayer and len(errors_batch.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the last ConvLayer, we have to resize the input
                errors_batch_hipster = errors_batch_hipster.reshape(-1, cur_out_hipster.shape[1], cur_out_hipster.shape[2], cur_out_hipster.shape[3])

            if type(layer) is ConvLayer and len(prev_out.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the first ConvLayer, we have to resize the prev_out (the image itself)
                prev_out_hipster = prev_out_hipster.reshape(-1, 3, 32, 32)

            # if type(layer) is FCLayer and len(prev_out.shape) > 3:
                # print("reshaping for FCLayer")
                # layer is the first FC after convolutional layers, we have to reshape prev_out
                # prev_out = prev_out.reshape(-1, prev_out.shape[1] * prev_out.shape[2] * prev_out.shape[3])

            # we skip the last output as it contains the final classification output
            (errors_batch_hipster, dW_hipster, db_hipster) = hl.conv_backward(errors_batch_hipster, cache_hipster[-i])

            grad_W_hipster[-i] = dW_hipster
            grad_b_hipster[-i] = db_hipster
            i += 1

            # print('backward_pass for', type_name, 'computed in {:6f}'.format(timer() - time))

        return grad_W_hipster, grad_b_hipster

    def backward_pass(self, errors_batch, outputs_batch):
        ''' do the backward pass and return grads for W update '''
        i = 1
        grad_W = len(self.layers) * [None]
        grad_b = len(self.layers) * [None]

        for layer in reversed(self.layers):

            if type(layer) is ConvLayer:
                type_name = "convlayer"
            else:
                type_name = "fclayer"
            time = timer()

            cur_out = np.array(outputs_batch[-i])
            prev_out = np.array(outputs_batch[-1 - i])

            if type(layer) is ConvLayer and len(errors_batch.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the last ConvLayer, we have to resize the input
                errors_batch = errors_batch.reshape(-1, cur_out.shape[1], cur_out.shape[2], cur_out.shape[3])

            if type(layer) is ConvLayer and len(prev_out.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the first ConvLayer, we have to resize the prev_out (the image itself)
                prev_out = prev_out.reshape(-1, 3, 32, 32)

            if type(layer) is FCLayer and len(prev_out.shape) > 3:
                print("reshaping for FCLayer")
                # layer is the first FC after convolutional layers, we have to reshape prev_out
                prev_out = prev_out.reshape(-1, prev_out.shape[1] * prev_out.shape[2] * prev_out.shape[3])

            # we skip the last output as it contains the final classification output
            (dW, db, errors_batch) = layer.backprop(errors_batch, cur_out, prev_out)

            # print(db_hipster.shape)

            grad_W[-i] = dW # we use the returned order here in order to obtain the regular order in the end
            grad_b[-i] = db # the same here

            i += 1

            # print('backward_pass for', type_name, 'computed in {:6f}'.format(timer() - time))

        return grad_W, grad_b

    def get_minibatch_grads(self, X, Y):
        ''' return gradients with respect to W and b for a minibatch '''
        # outputs_batch = x + layers outputs => len = nb_layers + 1
        # outputs_batch[layer_i] would be array of minibatch_size outputs of layer_i
        # we do the forward_pass for the whole batch at once
        # time = timer()
        outputs_batch = self.forward_pass(X)
        # for out_our, out_hipster in zip(outputs_batch, outputs_batch_hipster):
        #     error = out_our - out_hipster
        #     # error[error < 1e-3] = 0
        #     print("np.max(error) = ", np.max(error))
        #     print('hip_to_mine_conv_mistake = {}'.format(np.mean(error)))
        # print('forward_pass computed in {:6f}'.format(timer() - time))

        # we get the errors for the whole batch at once
        errors_batch = outputs_batch[-1] - Y
        loss = CE_loss_batch(Y, outputs_batch[-1])
        # print("get_minibatch_grads loss = ", loss)

        # time = timer()
        grads_W, grads_b = self.backward_pass(np.array(errors_batch), outputs_batch[:-1])
        # for grad_W_our, grad_b_our, grad_W_hipster, grad_b_hipster in zip(grads_W, grads_b, grads_W_hipster, grads_b_hipster):
        #     error_W = grad_W_our - grad_W_hipster
        #     error_b = grad_b_our - grad_b_hipster
        #     print("np.max(error_W) = ", np.max(error_W))
        #     print('hip_to_mine_grad_W_mistake = {}'.format(np.mean(error_W)))
        #     print("np.max(error_b) = ", np.max(error_b))
        #     print('hip_to_mine_grad_b_mistake = {}'.format(np.mean(error_b)))
        # print('backward_pass computed in {:6f}'.format(timer() - time))

        return grads_W, grads_b, loss


    def fit(self, X_train, y_train, K, minibatch_size, n_iter,
            X_cv = None, y_cv = None, step_size = 0.1, epsilon = 1e-8, gamma = 0.9, use_vanila_sgd = False):
        ''' train the network and adjust the weights during n_iter iterations '''

        # do the label preprocessing first
        y_train_vector = np.zeros((y_train.shape[0], K))
        for i in range(y_train.shape[0]):
            y_train_vector[i, y_train[i]] = 1

        # we will need this for RMSprop
        E_g_W = [np.zeros(layer.get_W_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. W
        E_g_b = [np.zeros(layer.get_b_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. b

        loss = math.inf
        # do fixed number of iterations
        for iter in range(n_iter):
            print("Iteration %d" % iter)
            X_train, y_train_vector = shuffle(X_train, y_train_vector)
            prev_loss = loss
            loss = 0
            proc_done = 0

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                X_minibatch = X_train[i:i + minibatch_size] # TODO: check if it is okey when X_size % minibatch_size != 0
                y_minibatch = y_train_vector[i:i + minibatch_size]

                (grads_W, grads_b, minibatch_loss) = self.get_minibatch_grads(X_minibatch, y_minibatch) # implement with the backward_pass

                # if i == 0 or i == 5000:
                    # print("grads_W = ", grads_W)
                    # print("W = ", [layer.W for layer in self.layers])

                loss += minibatch_loss

                # print("minibatch_loss = ", minibatch_loss)

                # update matrixes E_g_W and E_g_b used in the stepsize of RMSprop
                for j in range(self.nb_layers):
                    E_g_W[j] = gamma * E_g_W[j] + (1 - gamma) * (grads_W[j] ** 2)
                    E_g_b[j] = gamma * E_g_b[j] + (1 - gamma) * (grads_b[j] ** 2)

                # do gradient step for every layer
                for j in range(self.nb_layers):
                    if use_vanila_sgd:
                        self.layers[j].update(step_size * grads_W[j], step_size * grads_b[j])
                    else:
                        # do RMSprop step
                        self.layers[j].update(step_size / np.sqrt(E_g_W[j] + epsilon) * grads_W[j],
                                              grads_b[j] * step_size / np.sqrt(E_g_b[j] + epsilon))

                # if 1. * i / X_train.shape[0] > (proc_done + 1) * 0.1:
                    # print("%d out of %d done" % (i, X_train.shape[0]))
                    # proc_done += 1

            print("Loss = %f" % (loss / X_train.shape[0]))

            if X_cv is not None and y_cv is not None:
                y_cv_vector = np.zeros((y_cv.shape[0], K))
                for i in range(y_cv.shape[0]):
                    y_cv_vector[i, y_cv[i]] = 1
                y_cv_pred = self.predict(X_cv)
                accs = (y_cv_pred == y_cv).sum() / y_cv.size
                print("Accuracy on cross validation = %f" % accs)

            if np.absolute(loss - prev_loss) < np.sqrt(epsilon):
                print("Termination criteria is true, I stop the learning...")
                break

    def predict(self, X_test):
        ''' make prediction for all elements in X_test based on the learnt model '''
        y_test = []
        for X in X_test:
            prediction = np.argmax(self.forward_pass(X)[-1])
            y_test.append(prediction)
        return np.array(y_test)

def main():
    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1]
    # import pudb; pudb.set_trace()  # XXX BREAKPOINT
                
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)
                    
    ch1 = 12

    cnn = ConvNet()
    cnn.add_layer("convlayer", layer_info = {"in_channels": 3,
                                             "out_channels": ch1,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 1,
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

    out, cache =  hl.conv_forward(input_batch, W1, np.expand_dims(b1, 1), stride = 1, padding = 1)
    OUT = hl.conv_backward(out - er1, cache)

    print('hip_to_mine_conv_mistake = {}'.format(np.mean(V - out)))
    print('hip_to_mine_grad_mistake = {}'.format(np.mean(B[0] - OUT[1])))
    print('hip_to_mine_grad_sum  = {} {}'.format(np.sum(B[0]), np.sum(OUT[1])))
    # import pudb; pudb.set_trace()  # XXX BREAKPOINT

    sys.exit()

    # TENSORFLOW RELATED STUFF
    # INPUTS: batch|channels|height|width --> batch|height|width|channels
    tf_input_batch = input_batch.transpose(0, 2, 3, 1)
    # WEIGHTS: out_channels|in_channels|height|width ---> heigh|width|in_channels|out_channels
    tf_W1 = W1.transpose(2, 3, 1, 0)
    # LOSS: like inputs
    tf_er1 = er1.transpose(0, 2, 3, 1)

    import tensorflow as tf
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
    opt = tf.train.GradientDescentOptimizer(0.1)
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
   
   # Hipsternet


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
    cnn.fit(X, Y, X_cv = X_val, y_cv = Y_val, K = 2, minibatch_size = 50, n_iter = 30)

    # print(Y_train)
    # print(Y_test)

    y_pred = cnn.predict(X_test)
    accs = (y_pred == Y_test).sum() / Y_test.size
    print('Mean accuracy: %f' % accs)

def try_kaggle_fcnn():
    X_train_full = pd.read_csv('../data/Xtr.csv', header = None).as_matrix()[:,:-1]
    y_train_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
    X_test = pd.read_csv('../data/Xte.csv', header = None).as_matrix()[:,:-1]

    X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    # vis_img(X_test[40,:])

    print(X_train.shape)
    print(y_train.shape)
    print(X_cv.shape)
    print(y_cv.shape)
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

    cnn.fit(X_train, y_train, K = nb_classes, X_cv = X_cv, y_cv = y_cv, minibatch_size = 50, n_iter = 30)
    #y_test = cnn.predict(X_test)
    #y_test.dump("Yte.dat")

def test_poollayer():
    X_test = pd.read_csv('../data/Xte.csv', header = None).as_matrix()[:,:-1]
    cnn = ConvNet()
    cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"})

    cnn.layers[0].assert_pool_layer()
    # img_1 = X_test[0,:].reshape(3, 32, 32)
    # img_2 = X_test[4,:].reshape(3, 32, 32)

    # vis_img(X_test[40,:])

    # X_out = cnn.forward_pass(np.array([img_1, img_2]))[1]

    # print(X_out.shape)
    # X_out = X_out.reshape(2, -1)
    # X_out_1 = X_out[1]
    # img1 = X_out[0,:,:,:].transpose(1, 2, 0) + [0.25, 0.2, 0.2]
    # print(img1.shape)
    # plt.imshow(img1)

    # plt.show()
    # vis_img(X_test[40,:])
    # vis_img(X_out)

def prepro(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean

def test_convlayers():
    # get kaggle data
    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1].astype(int)

    # get CIFAR_orig data
    # data = unpickle("../cifar_orig/data_batch_5")
    # X_train_full = data[b'data']
    # y_train_full = np.array(data[b'labels'])

    # X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    # get MNIST data
    mnist = input_data.read_data_sets('../MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    nb_samples, data_length, nb_classes = X_train.shape[0], X_train.shape[1], y_train.max() + 1
    img_shape = (1, 28, 28)

    X_train, X_val, X_test = prepro(X_train, X_val, X_test)

    X_train = X_train.reshape(-1, *img_shape)
    X_val = X_val.reshape(-1, *img_shape)
    X_test = X_test.reshape(-1, *img_shape)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)


    ch1 = 32
    ch2 = 32
    ch3 = 64
    # ch4 = 8
    # ch5 = 8
    nb_classes = 10

    cnn = ConvNet()
    cnn.add_layer("convlayer", layer_info = {"in_channels": img_shape[0],
                                             "out_channels": ch1,
                                             "height": 5,
                                             "width": 5,
                                             "stride": 1,
                                             "padding": 1,
                                             "activation_type": "ReLU"}) # 25 x 25 x ch1
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch1,
                                             "out_channels": ch2,
                                             "height": 3,
                                             "width": 3,
                                             "stride": 2,
                                             "padding": 0,
                                            "activation_type": "ReLU"}) # 11 x 11 x ch2
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch2,
                                             "out_channels": ch3,
                                             "height": 3,
                                             "width": 3,
                                             "stride": 2,
                                             "padding": 0,
                                             "activation_type": "ReLU"}) # 4 x 4 x ch3
    cnn.add_layer("convlayer", layer_info = {"in_channels": ch3,
                                             "out_channels": nb_classes,
                                             "height": 4,
                                             "width": 4,
                                             "stride": 1,
                                             "padding": 0,
                                             "activation_type": "None"}) # 1 x 1 x 10
    cnn.fit(X_train, y_train, K = nb_classes, X_cv = X_val, y_cv = y_val, minibatch_size = 50, n_iter = 30, step_size=0.01, use_vanila_sgd=True)

def run_cnn(X_train, X_cv, y_train, y_cv):
    # X_train_full = pd.read_csv('../data/Xtr.csv', header = None).as_matrix()[:,:-1]
    # y_train_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
    # X_test = pd.read_csv('../data/Xte.csv', header = None).as_matrix()[:,:-1]

    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1]

    X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    # vis_img(X_test[40,:])

    print(X_train.shape)
    print(y_train.shape)
    print(X_cv.shape)
    print(y_cv.shape)
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
                                             "height": 6,
                                             "width": 6,
                                             "stride": 2,
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

    cnn.fit(X_train, y_train, K = nb_classes, X_cv = X_cv, y_cv = y_cv, minibatch_size = 50, n_iter = 30)

def fit_kaggle_data():
    X_train_full = pd.read_csv('../data/Xtr.csv', header = None).as_matrix()[:,:-1]
    y_train_full = pd.read_csv('../data/Ytr.csv').as_matrix()[:,1]
    X_test = pd.read_csv('../data/Xte.csv', header = None).as_matrix()[:,:-1]

    X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)

    # vis_img(X_test[40,:])

    run_cnn(X_train, X_cv, y_train, y_cv)

def whitening(data):
    nb_channels = 3
    channel_length = data.shape[1] / nb_channels
    # means = np.zeros(nb_chanels)
    # for d in data:
    #     for ch in range(nb_channels):
    #         for i in range(channel_length):
    #             means[ch] += d[ch * channel_length + i]
    # means /= data.shape[0] * channel_length

    # sigmas = np.zeros(nb_channels)
    # for d in data:
    #     for ch in range(nb_channels):
    #         for i in range(channel_length):
    #             sigmas += (d[ch * channel_length + i] - means[ch]) ** 2
    # sigmas /= data.shape[0] * channel_length

    # for ch in range(nb_channels):
    #     data[:, ch * channel_length: (ch + 1) * channel_length] = (data[:, ch * channel_length: (ch + 1) * channel_length] - means[ch]) / 
    # print("mean before ", data[:, 0:1024].var())

    r_data = whiten(data[:, 0:1024])
    g_data = whiten(data[:, 1024:2048])
    b_data = whiten(data[:, 2048:3072])

    # print("mean after ", r_data.var())
    # print(r_data)

    # features  = np.array([[1.9, 2.3, 1.7],[1.5, 2.5, 2.2],[0.8, 0.6, 1.7,]])
    # print("mean before ", features.var())
    # new_features = whiten(features)
    # print("mean after ", new_features.var())

    return np.concatenate((r_data, g_data, b_data), axis=1)


    # return data

def fit_orig_cifar():
    data = unpickle("../cifar_orig/data_batch_5")
    X = data[b'data']
    y = np.array(data[b'labels'])

    print(X.shape)

    X_white = whitening(X)

    print(X_white.shape)

    X_train, X_cv, y_train, y_cv = train_test_split(X_white, y, test_size = 0.05)

    run_cnn(X_train, X_cv, y_train, y_cv)

if __name__ == "__main__":
    # np.random.seed(500)
    # main()
    # test_moons()
    # try_kaggle_fcnn()
    # test_poollayer()
    # fit_kaggle_data() # check if it still works
    # fit_orig_cifar()
    # test_convlayers()



    











    

