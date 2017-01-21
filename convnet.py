import numpy as np

def ReLU(input_array):
    return np.maximum(input_array, 0)

def CE_loss(labels, outputs):
    # cross entropy loss
    loss = 0
    for i in range(len(labels)):
        loss += math.log(outputs[i, labels[i]])
    return loss


class fclayer:

    def __init__(self, input_size, output_size, activation_type):
        # self.n = n
        self.W = np.random.randn(output_size, input_size)

        # so far only ReLU is implemented
        if activation_type == "ReLU":
            self.activation = ReLU
        else:
            print("error: unknown activation type")
            self.activation = (lambda x: x)

    def forwardprop(self, X):
        out = W.dot(X)
        return self.activation(out)

    def backprop(self, grad, X):
        dW = dD.dot(X.T)
        dX = W.T.dot(dD)
        return smth

class convlayer:

    def __init__(self, nb_filters, stride, activation):
        self.n = n

    def forwardprop(self):

    def backprop(self):

class poollayer:

    def __init__(self):

class softmaxlayer:

    def __init__(self):


class convnet:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, layer_info):
        if type == "fclayer":
            layers.append(fclayer(layer_info))
            self.nb_layers += 1
        elif type == "convlayer":
            layers.append(convlayer(layer_info))
            self.nb_layers += 1
        elif if type == "poollayer":
            layers.append(poollayer(layer_info))
            self.nb_layers += 1
        else:
            print("error: unknown layer type")

    def forward_pass(self, X):
        cur_input = X
        for layer in self.layers:
            cur_input = layer.forwardprop(cur_input)
        return cur_input

    def fit(self, dataset, step_size):

    def predict(self):



