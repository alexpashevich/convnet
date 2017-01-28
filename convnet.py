import numpy as np
import pandas as pd
import pickle
import math
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.datasets import make_moons
#from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import tensorflow as tf
# import matplotlib.pyplot as plt


def ReLU(input_array):
    # rectified linear unit activation function
    return np.maximum(input_array, 0)

def CE_loss(y_true, y_pred):
    # cross entropy loss (multinomial regression)
    y_pred[y_pred < 1e-7] = 1e-7 # we clip values in order to prevent log of very small value
    return -np.sum(y_true * np.log(y_pred))

def sigmoid(input_array): # not used now
    # sigmoid activation function
    return 1. / (1 + np.exp(-input_array))

def square_loss(y_true, y_pred): # not used now
    # square loss
    return 1. / 2 * np.sum((y_true - y_pred) ** 2)


class fclayer:

    def __init__(self, layer_info):
        input_size = layer_info["input_size"]
        output_size = layer_info["output_size"]
        activation_type = layer_info["activation_type"]
        self.W = np.random.randn(input_size, output_size) * 0.01 # as in the AlexNet paper
        self.b = np.random.randn(output_size) * 0.01
        self.activation_type = activation_type # so far only ReLU is implemented

    def get_W_shape(self):
        return self.W.shape

    def get_b_shape(self):
        return self.b.shape

    def forwardprop(self, X):
        out = X @ self.W + self.b

        if self.activation_type == "ReLU":
            out = ReLU(out)
        elif self.activation_type == "None":
            out = out # no activation
        else:
            print("error: unknown activation type")
            out = out
        return out

    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        if self.activation_type == "ReLU":
            error_batch[cur_out_batch <= 0] = 0

        dW = prev_out_batch.T @ error_batch
        db = np.sum(error_batch, axis=0)
        dA = error_batch @ self.W.T

        return dW, db, dA

    def update(self, update_W, update_b):
        self.W -= update_W
        self.b -= update_b

"""
im2col trick
Courtesy of :
    https://github.com/wiseodd/hipsternet/blob/f4b46e0a7856e45553955893b266df60bae8083c/hipsternet/im2col.py
"""
def get_im2col_indices(in_channels, height, width, out_height, out_width, stride):
    i0 = np.repeat(np.arange(height), width) 
    i0 = np.tile(i0, in_channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(width), height * in_channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(in_channels), height * width).reshape(-1, 1)
    k = k.astype(int)
    i = i.astype(int)
    j = j.astype(int)
    return k, i, j

class ConvLayer(object):
    def __init__(self, W, b, stride=1, padding=0):   
        """
        W - [out_channels, in_channels, height, width]
        """
        self.W = W
        self.b = b
        self.out_channels, self.in_channels, self.height, self.width = W.shape
        self.stride = stride
        self.padding = padding

    def slow_fprop(self, X, out_height, out_width): # let it just be here for some time
        time = timer()
        batch_size = X.shape[0]
        output = np.zeros([batch_size, self.out_channels, out_height, out_width])
        s = self.stride
        for b in range(0, batch_size):
            for k in range(0, self.out_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        output[b, k, i, j] = np.add(
                                np.sum(np.multiply(
                                        X[b, :, (i*s):(i*s+self.height), (j*s):(j*s+self.width)],
                                        self.W[k, :, :, :]
                                        )), 
                                self.b[k]
                                )
        print('Slow thing computed in {:6f}'.format(timer() - time))
        return output

    def fast_fprop(self, X, out_height, out_width):
        time = timer()
        batch_size = X.shape[0]
        in_channels = X.shape[1]
        k, i, j = get_im2col_indices(self.in_channels, self.height, self.width, out_height, out_width, self.stride)
        X_col = X[:, k, i, j]  # (batch_size)*(H*W*in_channels)x(oH*oW)
        X_col = X_col.transpose(1, 2, 0).reshape(self.height * self.width * in_channels, -1) 
        W_col = self.W.reshape(self.out_channels, -1)
        output = np.matmul(W_col, X_col) + np.expand_dims(self.b, 1)
        output = output.reshape(self.out_channels, out_height, out_width, batch_size).transpose(3, 0, 1, 2)
        print('Fast thing computed in {:6f}'.format(timer() - time))
        return output
        
    def forwardprop(self, X, slow=False):
        """
        X - [batch_size, in_channels, in_height, in_width]
        """
        batch_size, in_channels, in_height, in_width = X.shape
        out_height, out_width = self.get_output_dims(X)
        
        p = self.padding
        if p == 0:
            Xp = X
        else:
            Xp = np.pad(X, ((0, 0), (p, p), (p, p)), mode='constant')
        
        # Vs = self.slow_fprop(Xp, out_height, out_width)
        output = self.fast_fprop(Xp, out_height, out_width)
        # print(np.mean(Vs-V))
        
#        if self.activation_type == "ReLU":
#            out = ReLU(V)
#        elif self.activation_type == "None":
#            out = out # no activation
#        else:
#            print("error: unknown activation type")
#            out = out
        #return ReLU(output)
        return output


    def get_output_dims(self, X): # we need it because not every filter size can be applied
        batch_size, in_channels, in_height, in_width = X.shape
        assert in_channels == self.in_channels
        assert (in_height - self.height + 2*self.padding) % self.stride == 0
        assert (in_width - self.width + 2*self.padding) % self.stride == 0
        out_height = (in_height - self.height + 2*self.padding) // self.stride + 1
        out_width = (in_width - self.width + 2*self.padding) // self.stride + 1
        return out_height, out_width
        
    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        """
        What shall we do here:
        1. Compute derivative of activ. func. in cur_out_batch
        2. Multiply transposed prev. weights (prev_out_batch) by error batch
        3. Multiply 1 by 2              
        """
      #  if self.activation_type == "ReLU":
        #error_batch[cur_out_batch <= 0] = 0 # Step 1

        X = prev_out_batch # previous output of the layer
        batch_size, in_channels, in_height, in_width = X.shape
        out_height, out_width = self.get_output_dims(X)
        #import pudb; pudb.set_trace()  # XXX BREAKPOINT
        
        # Do im2col trick once again
        k, i, j = get_im2col_indices(self.in_channels, self.height, self.width, out_height, out_width, self.stride)
        X_col = X[:, k, i, j]  # (batch_size)*(H*W*in_channels)x(oH*oW)
        X_col = X_col.transpose(1, 2, 0).reshape(self.height * self.width * in_channels, -1) 
        # Here we just transposed X into columns, in the same way as in forward phase
        
        # here we sum up all errors, reshape them into matrix as well
        db = np.sum(error_batch, axis=(0, 2, 3)) # ?? Do we sum it as it should be
        #db = np.sum(error_batch, axis=(0, 3, 2))
        db = db.reshape(self.out_channels, -1) 
        
        # Here - we reshape batch of errors in order to multiply it by weights
        dout_reshaped = error_batch.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        dW = np.matmul(dout_reshaped, X_col.T)
        dW = dW.reshape(self.W.shape)
        W_reshape = self.W.reshape(self.out_channels, -1)
        dX_col = W_reshape.T @ dout_reshaped
        
        # Reshape dX back
        dX_reshaped = dX_col.reshape(self.in_channels * self.height * self.width, -1, batch_size).transpose(2, 0, 1)
        h_pad, w_pad = in_height + 2*self.padding, in_width + 2*self.padding
        x_pad = np.zeros((batch_size, self.in_channels, h_pad, w_pad), dtype = dX_col.dtype)
        np.add.at(x_pad, (slice(None), k, i, j), dX_reshaped)
        # remove padding (if any)
        if self.padding == 0:
            dX = x_pad
        else:
            dX = x_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
       
        return dW, db, dX
        
# class poollayer:
#     # TODO

#     def __init__(self):

#     def forwardprop(self):

#     def backprop(self):

class ConvNet:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, layer_type, layer_info):
        ''' add a layer to the NN '''
        if layer_type == "fclayer":
            self.layers.append(fclayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "convlayer":
            self.layers.append(convlayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "poollayer":
            self.layers.append(poollayer(layer_info))
            self.nb_layers += 1
        else:
            print("error: unknown layer type")

    def forward_pass(self, X):
        ''' return the input data X and outputs of every layer '''
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        for layer in self.layers:
            cur_input = layer.forwardprop(cur_input)
            outputs.append(cur_input)

        # the softmax layer, we subtract maximum to avoid overflow
        cur_input = np.exp(cur_input - np.max(cur_input)) / np.exp(cur_input - np.max(cur_input)).sum()
        outputs.append(cur_input)

        return outputs

    def backward_pass(self, errors_batch, outputs_batch):
        ''' do the backward pass and return grads for W update '''
        i = 1
        grad_W = len(self.layers) * [None]
        grad_b = len(self.layers) * [None]
        for layer in reversed(self.layers):
            # we skip the last output as it contains the final classification output
            (dW, db, errors_batch) = layer.backprop(errors_batch, np.array(outputs_batch[-i]), np.array(outputs_batch[-1 - i]))
            grad_W[-i] = dW # we use the returned order here in order to obtain the regular order in the end
            grad_b[-i] = db # the same here
            i += 1

        return grad_W, grad_b

    def get_minibatch_grads(self, X, Y):
        ''' return gradients with respect to W and b for a minibatch '''
        loss = 0
        errors_batch = []
        # outputs = x + layers outputs => len = nb_layers + 1
        # outputs_batch[layer_i] would be array of minibatch_size outputs of layer_i
        outputs_batch = [[] for _ in range(self.nb_layers + 1)]

        # we do the forward_pass and stack all the outputs in the right order
        for x, y in zip(X, Y):
            outputs = self.forward_pass(x)

            errors = outputs[-1] - y
            # TODO: redo everything with fancy numpy functions instead of ugly loops
            for i in range(self.nb_layers + 1):
                outputs_batch[i].append(outputs[i])
            errors_batch.append(errors)
            loss += CE_loss(y, outputs[-1])

        grads_W, grads_b = self.backward_pass(np.array(errors_batch), outputs_batch)

        return grads_W, grads_b, loss


    def fit(self, X_train, y_train, K, minibatch_size, n_iter,
            X_cv = None, y_cv = None, step_size = 0.01, epsilon = 1e-8, gamma = 0.9, use_vanila_sgd = False):
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

        
def get_data_fast(name):
    #some problems with training labels, fix later
    data_csv_path = Path('.').resolve().parent/"Data"/(name + ".csv")
    data_pkl_path = data_csv_path.parent/(name+".pkl")
    f = None
    try:
        with data_pkl_path.open('rb') as f:
            data = pickle.load(f)
    except (OSError, IOError) as e:
        f = str(data_csv_path)
        data = np.genfromtxt(fname = str(data_csv_path), delimiter = ",")
        with data_csv_path.open('wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    #data = data[:,:-1]
    return data

def vis_img(x):
    """ Take image of dims [channels, h, w], show"""
    img = x.reshape(3, 32, 32).transpose(1, 2, 0) + [0.25, 0.2, 0.2]
    plt.imshow(img)
    plt.show()
    

def main():
    # Playing with data
    xtr = get_data_fast("Xtr")[:,:-1]
    # xte = get_data_fast("Xte")[:,:,-1]
    # ytr = get_data_fast("Ytr")
    # Traspose img to [batch_size, channels, im_height, im_width]
    # Building net
    xtr = xtr.reshape(-1, 3, 32, 32)
    W1 = np.random.normal(0, 1, (12, 3, 5, 5)) # random weights, 12 filters
    b1 = np.ones([12]) * 0.1
    cv1 = ConvLayer(W1, b1)
    #
    input_batch = xtr[:3, :, :, :]
    # Doing forward pass
    V = cv1.forwardprop(input_batch) 
    er1 = np.random.random(V.shape)
    B = cv1.backprop(V-er1, V, input_batch)


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
    opt = tf.train.AdagradOptimizer(0.1)
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
    import sys
    sys.path.append('/media/d/study/Grenoble/courses/advanced_learning_models/Competition/temp/hipsternet')
    import hipsternet.layer as hl
    out, cache =  hl.conv_forward(input_batch, W1, np.expand_dims(b1, 1), stride = 1, padding = 0)
    OUT = hl.conv_backward(out - er1, cache)

    print('hip_to_mine_conv_mistake = {}'.format(np.mean(V - out)))
    print('hip_to_mine_grad_mistake = {}'.format(np.mean(B[0] - OUT[1])))
    print('hip_to_mine_grad_sum  = {} {}'.format(np.sum(B[0]), np.sum(OUT[1])))
    

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


if __name__ == "__main__":
    np.random.seed(500)
    main()
    # test_moons()
    # try_kaggle()


    











    

