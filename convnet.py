import numpy as np
import pickle
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.datasets import load_iris, make_moons
from sklearn.cross_validation import train_test_split
from timeit import default_timer as timer
import matplotlib.pyplot as plt

#from yann import run_nn

def ReLU(input_array):
    # rectified linear unit activation function
    return np.maximum(input_array, 0)

def CE_loss(labels_x, output_x):
    # cross entropy loss (multinomial regression)
    return -np.sum(labels_x * np.log(output_x))

def sigmoid(input_array): # not used now
    # sigmoid activation function
    return 1. / (1 + np.exp(-input_array))

def square_loss(labels_x, output_x): # not used now
    # square loss
    return 1. / 2 * np.sum((labels_x - output_x) ** 2)


class fclayer:

    def __init__(self, layer_info):
        input_size = layer_info["input_size"]
        output_size = layer_info["output_size"]
        activation_type = layer_info["activation_type"]
        self.W = np.random.randn(output_size, input_size) # * 0.01 # as in the AlexNet paper
        self.b = np.random.randn(output_size) # * 0.01
        self.activation_type = activation_type # so far only ReLU is implemented

    def get_W_shape(self):
        return self.W.shape

    def get_b_shape(self):
        return self.b.shape

    def forwardprop(self, X):
        # print("W_shape = ", self.W.shape)
        # print("b_shape = ", self.b.shape)
        # print("X_shape = ", X.shape)
        out = self.W.dot(X) + self.b

        # print("out_shape = ", out.shape)

        if self.activation_type == "ReLU":
            out = ReLU(out)
        elif self.activation_type == "None":
            out = out # no activation
        else:
            print("error: unknown activation type")
            out = out
        return out

    def backprop(self, error, cur_out, prev_out):
        # dW = error.dot(prev_out.T) # this should be an array with shape (output_size, input_size)
        dW = np.outer(error, prev_out)
        if self.activation_type == "ReLU":
            dW[cur_out < 0] = 0

        db = error
        # print("db_shape = ", db.shape)
        if self.activation_type == "ReLU":
            db[cur_out < 0] = 0

        dA = self.W.T.dot(error) # this should be a vector with length input_size
        if self.activation_type == "ReLU":
            dA[cur_out < 0] = 0

        # in the example it was
        # dW = dD.dot(X.T)
        # dX = W.T.dot(dD)

        return dW, db, dA

    def update(self, update_W, update_b):
        self.W += update_W
        self.b += update_b


class ConvLayer(object):
    def __init__(self, W, b, stride=1, padding=0):   
        """
        W - [height, width, in_channels, out_channels]
        """
        self.W = W
        self.b = b
        self.height, self.width, self.in_channels, self.out_channels = W.shape
        self.stride = stride
        self.padding = padding

        
    def forwardprop(self, X):
        in_height, in_width, in_channels = X.shape
        out_height, out_width = self.get_output_dims(X)

        V = np.zeros([out_height, out_width, self.out_channels])
        s = self.stride

        # Slow thing
        time = timer()
        for i in range(0, out_height):
            for j in range(0, out_width):
                for k in range(0, self.out_channels):
                    V[i,j,k] = np.sum(
                            X[
                                (i*s):(i*s+self.height), 
                                (j*s):(j*s+self.width), :
                                ]*self.W[:, :, :, k] + self.b[k]
                            )
        print('Slow thing computed in {:6f}'.format(timer() - time))

        try_fast = False 

        if (try_fast):

            ########################################################
            ######  Well, this should work fast, but later      ####
            ########################################################
            
            # Will do a separate procedure, it's too long
            # First of all we introduce super fancy indixes

            i0 = np.repeat(np.arange(self.height), self.width) 
            i0 = np.tile(i0, self.in_channels)
            i1 = self.stride * np.repeat(np.arange(out_height), out_width)
            j0 = np.tile(np.arange(self.width), self.height * self.in_channels)
            j1 = self.stride * np.tile(np.arange(out_width), out_height)
            i = i0.reshape(-1, 1) + i1.reshape(1, -1)
            j = j0.reshape(-1, 1) + j1.reshape(1, -1)
            k = np.repeat(np.arange(self.in_channels), self.height * self.width).reshape(-1, 1)
            k = k.astype(int)
            i = i.astype(int)
            j = j.astype(int)
            
#            # now we transform our data according to the indexes and get a vector instead of a tensor
#            # not sure about it yet though [indexes and stuff]
#            x_padded = np.zeros((N, in_channels, (in_heigth + 2 * padding), (in_width + 2 * padding)), dtype=cols.dtype)
#            cols = x_padded[:, k, i, j]
#            cols = cols.transpose(1, 2, 0).reshape(self.height * self.width * self.in_channels, -1)
#
#            # Do quick computations (with cool matrix multiplication)
#            
#            filter_col = self.filter.reshape(out_channels, -1)
#            out = filter_col @ X_col + b
#            out = out.reshape(out_channels, h_out, w_out, n_x)
#            out = out.transpose(3, 0, 1, 2)
#
#            # now we return  our vectors back to normal life
#
#            cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
#            cols_reshaped = cols_reshaped.transpose(2, 0, 1) # not really sure about indexes here
#            np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
#            if padding == 0:
#                x_original = x_padded
#            else:
#                x_original = x_padded[:, :, padding:-padding, padding:-padding]
#

        #import pudb; pudb.set_trace()  # XXX BREAKPOINT
        #pass
        
#        if self.activation_type == "ReLU":
#            out = ReLU(V)
#        elif self.activation_type == "None":
#            out = out # no activation
#        else:
#            print("error: unknown activation type")
#            out = out
        return ReLU(V)


    def get_output_dims(self, X):
        in_height, in_width, in_channels = X.shape

        assert in_channels == self.in_channels
        assert (in_height - self.height + 2*self.padding) % self.stride == 0
        assert (in_width - self.width + 2*self.padding) % self.stride == 0

        out_height = (in_height - self.height + 2*self.padding) // self.stride + 1
        out_width = (in_width - self.width + 2*self.padding) // self.stride + 1
        return out_height, out_width
        
    def backprop(self):
        # TODO
        pass

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
        # return the input data X and outputs of every layer
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        for layer in self.layers:
            cur_input = layer.forwardprop(cur_input)
            outputs.append(cur_input)

        # the softmax layer
        cur_input = np.exp(cur_input) / np.exp(cur_input).sum()
        outputs.append(cur_input)

        return outputs

    def backward_pass(self, labels, outputs):
        # do the backward pass and return grads for W update
        i = 1
        grad_W = len(self.layers) * [None]
        grad_b = len(self.layers) * [None]
        errors = outputs[-1] - labels # we expect CE loss and softmax in the end
        for layer in reversed(self.layers):
            # we skip the last output as it contains the final classification output
            (dW, db, errors) = layer.backprop(errors, cur_out=outputs[-1 - i], prev_out=outputs[-2 - i])
            grad_W[-i] = dW # we use the returned order here in order to obtain the normal order in the end
            grad_b[-i] = db # the same here
            i += 1

        return grad_W, grad_b

    def get_minibatch_grads(self, X, Y):
        # return averaged gards over the minibatch
        list_grads_W = []
        list_grads_b = []
        loss = 0
        minibatch_size = X.shape[0]
        for i in range(minibatch_size):
            outputs = self.forward_pass(X[i,:])
            grad_W, grad_b = self.backward_pass(Y[i,:], outputs)
            list_grads_W.append(grad_W)
            list_grads_b.append(grad_b)
            loss += CE_loss(Y[i,:], outputs[-1])

        grads_W = []
        grads_b = []
        for i in range(self.nb_layers):
            grads_W.append(np.zeros(self.layers[i].get_W_shape()))
            grads_b.append(np.zeros(self.layers[i].get_b_shape()))
            for j in range(minibatch_size):
                grads_W[i] += list_grads_W[j][i]
                grads_b[i] += list_grads_b[j][i]
            grads_W[i] /= minibatch_size
            grads_b[i] /= minibatch_size

        return grads_W, grads_b, loss


    def fit(self, X_train, Y_train, K, step_size, minibatch_size, n_iter):
        # train the network and adjust the weights during n iterations

        # do the label preprocessing first
        Y_train_vector = np.zeros((Y_train.shape[0], K))
        for i in range(Y_train.shape[0]):
            Y_train_vector[i, Y_train[i]] = 1

        # do fixed number of iterations
        for iter in range(n_iter):
            print("Iteration %d" % iter)
            X_train, Y_train_vector = shuffle(X_train, Y_train_vector)
            loss = 0

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                X_minibatch = X_train[i:i + minibatch_size] # TODO: check if it is okey when X_size % minibatch_size != 0
                Y_minibatch = Y_train_vector[i:i + minibatch_size]

                (grads_W, grads_b, minibatch_loss) = self.get_minibatch_grads(X_minibatch, Y_minibatch) # implement with the backward_pass
                loss += minibatch_loss
                # do gradient step for every layer
                # so far the step size is fixed, smth like RMSprop should be used ideally
                for i in range(self.nb_layers):
                    self.layers[i].update(-step_size * grads_W[i], -step_size * grads_b[i])

            print("Loss = %f" % (loss / X_train.shape[0]))

    def predict(self, X_test):
        # TODO: smth like this, check
        Y_test = []
        for X in X_test:
            prediction = np.argmax(self.forward_pass(X)[-1])
            Y_test.append(prediction)
        return Y_test

if __name__ == "__main__":
    # iris = load_iris()
    # X, Y = iris.data, iris.target
    # K = 3

    # size1 = X.shape[1]
    # size2 = 50
    # size3 = 40
    # size4 = K

    # cnn = ConvNet()
    # cnn.add_layer("fclayer", layer_info = {"input_size": size1, "output_size": size2, "activation_type": "ReLU"})
    # cnn.add_layer("fclayer", layer_info = {"input_size": size2, "output_size": size3, "activation_type": "ReLU"})
    # cnn.add_layer("fclayer", layer_info = {"input_size": size3, "output_size": size4, "activation_type": "ReLU"})

    # cnn.fit(X, Y, K = K, step_size = 0.01, minibatch_size = 10, n_iter = 500)
    
    #############################################
    # Real data, comment/uncomment if necessary
    #############################################
    
    #xtr_csv = np.genfromtxt(fname = str(xtr_csv_path), delimiter = ",")
    
    #with xtr_pkl_path.open('wb') as f:
    #   pickle.dump(xtr_csv, f, pickle.HIGHEST_PROTOCOL)
     
     # Should be a separate procedure later
    xtr_csv_path = Path('.').resolve().parent/"Data"/"Xtr.csv"
    xtr_pkl_path = xtr_csv_path.parent/'Xtr.pkl'
    try:
        with xtr_pkl_path.open('rb') as f:
            xtr = pickle.load(f)
    except (OSError, IOError) as e:
        xtr_csv = np.genfromtxt(fname = str(xtr_csv_path), delimiter = ",")
        pickle.dump(xtr_csv, f, pickle.HIGHEST_PROTOCOL)
 
    xtr = xtr[:,:-1]
    img = xtr[49].reshape(3, 32, 32).transpose(1, 2, 0)# + [0.25, 0.2, 0.2]
    cnn = ConvNet()
    # Building net
    W1 = np.random.normal(0, 1, (5, 5, 3, 12)) # random weights, 12 filters
    b1 = np.ones([12]) * 0.1
    cv1 = ConvLayer(W1, b1)
    # Doing forward pass
    V = cv1.forwardprop(img) # just to make sure I didn't mess up indices and all
    plt.imshow(V[:, :, 3])
    #####################################################


    X, Y = make_moons(n_samples=5000, random_state=42, noise=0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    # run_nn(X_train, X_test, Y_train, Y_test)
    size1 = 2
    size2 = 100
    size3 = 2

    cnn = ConvNet()
    cnn.add_layer("fclayer", layer_info = {"input_size": size1, "output_size": size2, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size2, "output_size": size3, "activation_type": "ReLU"})
    cnn.fit(X, Y, K = 2, step_size = 1e-4, minibatch_size = 50, n_iter = 100)












