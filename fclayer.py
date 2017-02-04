import numpy as np
from utils import ReLU


class FCLayer(object):

    def __init__(self, layer_info):
        self.input_size = layer_info["input_size"]
        self.output_size = layer_info["output_size"]
        self.activation_type = layer_info["activation_type"] # so far only ReLU is implemented
        self.W = np.random.randn(self.input_size, self.output_size) * 0.01 # as in the AlexNet paper
        self.b = np.random.randn(self.output_size) * 0.01

    def get_W_shape(self):
        return self.W.shape

    def get_b_shape(self):
        return self.b.shape

    def forwardprop(self, X):
        ''' X - [batch_size, input_size] '''
        # print("[FCLayer] X.shape = ", X.shape)

        out = X @ self.W + np.outer(np.ones(X.shape[0]), self.b)

        if self.activation_type == "ReLU":
            out = ReLU(out)
        elif self.activation_type == "None":
            out = out # no activation
        else:
            print("error: unknown activation type")
            out = out
        # print("[FCLayer] output.shape = ", out.shape)
        return out

    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        # print("[FCLayer_back] error_batch.shape = ", error_batch.shape)
        # print("[FCLayer_back] cur_out_batch.shape = ", cur_out_batch.shape)
        # print("[FCLayer_back] prev_out_batch.shape = ", prev_out_batch.shape)

        if self.activation_type == "ReLU":
            error_batch[cur_out_batch <= 0] = 0

        dW = prev_out_batch.T @ error_batch
        db = np.sum(error_batch, axis=0)
        dA = error_batch @ self.W.T

        return dW, db, dA

    def update(self, update_W, update_b):
        self.W -= update_W
        self.b -= update_b
