import numpy as np
from sklearn.datasets import load_iris

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

    def __init__(self, layer_info): #input_size, output_size, activation_type):
        input_size = layer_info["layer_info"]
        output_size = layer_info["output_size"]
        activation_type = layer_info["activation_type"]
        self.W = np.random.randn(output_size, input_size) * 0.01 # as in the AlexNet paper
        self.activation_type = activation_type # so far only ReLU is implemented

    def forwardprop(self, X):
        out = W.dot(X)

        if self.activation_type == "ReLU":
            out = ReLU(out)
        elif self.activation_type == "None":
            out = out # no activation
        else:
            print("error: unknown activation type")
            out = out
        return out

    def backprop(self, error, cur_out, prev_out):
        dW = error.dot(prev_out.T) # this should be an array with shape (output_size, input_size)
        if self.activation_type == "ReLU":
            dW[cur_out < 0] = 0

        dA = self.W.T.dot(error) # this should be a vector with length input_size
        if self.activation_type == "ReLU":
            dA[cur_out < 0] = 0

        # dW = dD.dot(X.T)
        # dX = W.T.dot(dD)
        return dW, dA

    def update(self, update_size):
        self.W += update_size

# class convlayer:
#     # TODO

#     def __init__(self, nb_filters, stride, activation_type):
#         # self.n = n

#     def forwardprop(self):

#     def backprop(self):

# class poollayer:
#     # TODO

#     def __init__(self):

#     def forwardprop(self):

#     def backprop(self):

# class softmaxlayer:

#     def __init__(self):

#     def forwardprop(self, X):
#         return np.exp(X) / np.exp(X).sum()

#     def backprop(self, smth):
#         # TODO ?

class ConvNet:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, layer_type, layer_info):
        if layer_type == "fclayer":
            layers.append(fclayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "convlayer":
            layers.append(convlayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "poollayer":
            layers.append(poollayer(layer_info))
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
        return outputs

    def backward_pass(self, X):
        # TODO
        a = 5

    def get_minibatch_grads(self, X_minibatch, Y_minibatch):
        # TODO
        a = 5

    def fit(self, X_train, Y_train, step_size, minibatch_size, n_iter):
        # do fixed number of iterations
        for iter in range(n_iter):
            print("Iteration %d" % iter)
            X_train, Y_train = shuffle(X_train, Y_train)

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                X_minibatch = X_train[i:i + minibatch_size]
                Y_minibatch = Y_train[i:i + minibatch_size]

                grads = get_grads(X_minibatch, Y_minibatch) # implement with the backward_pass

                # do gradient step for every layer
                # so far the step size is fixed, smth like RMSprop should be used ideally
                for i in range(self.nb_layers):
                    self.layers[i].update(step_size * grads[i])

    def predict(self, X_test):
        # smth like this, check
        Y_test = []
        for X in X_test:
            prediction = np.argmax(self.forward_pass(X)[-1])
            Y_test.append(prediction)
        return Y_test

if __name__ == "__main__":
    iris = load_iris()
    X, Y = iris.data, iris.target

    print(X.shape)
    print(Y.shape)

    size1 = X.shape[0]
    size4 = Y.shape[0]

    # cnn = ConvNet()
    # ConvNet.add_layer("fclayer", {"input_size": size1, "output_size": size2, "activation_type": ReLU})
    # ConvNet.add_layer("fclayer", {"input_size": size2, "output_size": size3, "activation_type": ReLU})
    # ConvNet.add_layer("fclayer", {"input_size": size3, "output_size": size4, "activation_type": ReLU})












