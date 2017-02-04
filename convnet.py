import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
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

    def forward_pass(self, X):
        ''' return the input data X and outputs of every layer '''
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        for layer in self.layers:
            if type(layer) is ConvLayer and len(cur_input.shape) < 4:
                # X -> ConvLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, 3, 32, 32)

            if type(layer) is FCLayer and len(cur_input.shape) > 2:
                # ConvLayer -> FCLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, cur_input.shape[1] * cur_input.shape[2] * cur_input.shape[3])

            cur_input = layer.forwardprop(cur_input)
            outputs.append(cur_input)

        # the softmax layer, we subtract maximum to avoid overflow
        cur_input = np.exp(cur_input - np.max(cur_input)) / np.outer(np.exp(cur_input - np.max(cur_input)).sum(axis=1), np.ones(cur_input.shape[1]))
        outputs.append(cur_input)

        return outputs

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
                # print("reshaping for FCLayer")
                # layer is the first FC after convolutional layers, we have to reshape prev_out
                prev_out = prev_out.reshape(-1, prev_out.shape[1] * prev_out.shape[2] * prev_out.shape[3])

            # we skip the last output as it contains the final classification output
            (dW, db, errors_batch) = layer.backprop(errors_batch, cur_out, prev_out)
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
        time = timer()
        outputs_batch = self.forward_pass(X)
        # print('forward_pass computed in {:6f}'.format(timer() - time))

        # we get the errors for the whole batch at once
        errors_batch = outputs_batch[-1] - Y
        loss = CE_loss_batch(Y, outputs_batch[-1])

        time = timer()
        grads_W, grads_b = self.backward_pass(np.array(errors_batch), outputs_batch[:-1])
        # print('backward_pass computed in {:6f}'.format(timer() - time))
        return grads_W, grads_b, loss


    def fit(self, X_train, y_train, K, minibatch_size, n_iter,
            X_cv = None, y_cv = None, step_size = 0.001, epsilon = 1e-8, gamma = 0.9, use_vanila_sgd = False):
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

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                X_minibatch = X_train[i:i + minibatch_size] # TODO: check if it is okey when X_size % minibatch_size != 0
                y_minibatch = y_train_vector[i:i + minibatch_size]

                (grads_W, grads_b, minibatch_loss) = self.get_minibatch_grads(X_minibatch, y_minibatch) # implement with the backward_pass

                loss += minibatch_loss

                # update matrixes E_g_W and E_g_b used in the stepsize of RMSprop
                for i in range(self.nb_layers):
                    E_g_W[i] = gamma * E_g_W[i] + (1 - gamma) * (grads_W[i] ** 2)
                    E_g_b[i] = gamma * E_g_b[i] + (1 - gamma) * (grads_b[i] ** 2)

                # do gradient step for every layer
                for i in range(self.nb_layers):
                    if use_vanila_sgd:
                        self.layers[i].update(step_size * grads_W[i], step_size * grads_b[i])
                    else:
                        # do RMSprop step
                        self.layers[i].update(step_size / np.sqrt(E_g_W[i] + epsilon) * grads_W[i],
                                              step_size / np.sqrt(E_g_b[i] + epsilon) * grads_b[i])
            print("Mean value of the gradient = %f" % (np.mean(grads_W[i])))
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
    sys.path.append('/media/d/study/Grenoble/courses/advanced_learning_models/Competition/temp/hipsternet')
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
        # with tf.device('/cpu:0'):
        with tf.device('/GPU:0'):
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

def try_kaggle():
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

    #cnn.fit(X_train, y_train, K = nb_classes, X_cv = X_cv, y_cv = y_cv, minibatch_size = 50, n_iter = 30)
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
                
    X_train, X_cv, y_train, y_cv = train_test_split(X_train_full, y_train_full, test_size = 0.1)
                    
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_cv.shape)
    # print(y_cv.shape)
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

    cnn.fit(X_train, y_train, K = nb_classes, X_cv = X_cv, y_cv = y_cv, minibatch_size = 50, n_iter = 30)


if __name__ == "__main__":
    # np.random.seed(500)
    main()
    # test_moons()
    # try_kaggle()
    # test_poollayer()
    # test_cnn()



    











    

