import numpy as np
from utils import ReLU


class FCLayer(object):

    def __init__(self, layer_info):
        self.input_size = layer_info["input_size"]
        self.output_size = layer_info["output_size"]
        self.activation_type = layer_info["activation_type"] # so far only ReLU is implemented
        if "W" in layer_info:
            self.W = layer_info["W"]
        else:
            self.W = np.random.randn(self.input_size, self.output_size) * 0.01 # as in the AlexNet paper
        if "b" in layer_info:
            self.b = layer_info["b"]
        else:
            self.b = np.random.randn(self.output_size) * 0.01


    def get_W_shape(self):
        return self.W.shape

    def get_b_shape(self):
        return self.b.shape

    def forwardprop(self, X):
        ''' X - [batch_size, input_size] '''

        out = X @ self.W + np.outer(np.ones(X.shape[0]), self.b)

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

    def dump_layer_info(self):
        dict_layer = {"input_size": self.input_size,
                      "output_size": self.output_size,
                      "activation_type": self.activation_type,
                      "W": self.W,
                      "b": self.b}
        layer_info = ["fclayer", dict_layer]
        return layer_info

    def get_layer_description(self):
        return 'FCL {} ({})'.format(self.output_size, self.activation_type)









