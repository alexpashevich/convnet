import numpy as np
from timeit import default_timer as timer
from utils import get_im2col_indices, ReLU

class ConvLayer(object):
    def __init__(self, layer_info): 
        # W - [out_channels, in_channels, height, width]
        self.in_channels = layer_info["in_channels"]
        self.out_channels = layer_info["out_channels"]
        self.height = layer_info["height"]
        self.width = layer_info["width"]
        self.stride = layer_info["stride"]
        self.padding = layer_info["padding"]
        self.activation_type = layer_info["activation_type"] # so far only ReLU is implemented
        if "W" in layer_info:
            self.W = layer_info["W"]
        else:
            self.W = np.random.randn(self.out_channels, self.in_channels, self.height, self.width) * 0.01
        if "b" in layer_info:
            self.b = layer_info["b"]
        else:
            self.b = np.random.randn(self.out_channels) * 0.01
        
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
        # time = timer()
        batch_size = X.shape[0]
        in_channels = X.shape[1]
        k, i, j = get_im2col_indices(self.in_channels, self.height, self.width, out_height, out_width, self.stride)
        X_col = X[:, k, i, j]  # (batch_size)*(H*W*in_channels)x(oH*oW)
        X_col = X_col.transpose(1, 2, 0).reshape(self.height * self.width * in_channels, -1) 
        W_col = self.W.reshape(self.out_channels, -1)
        output = np.matmul(W_col, X_col) + np.expand_dims(self.b, 1)
        output = output.reshape(self.out_channels, out_height, out_width, batch_size).transpose(3, 0, 1, 2)
        # print('Fast thing computed in {:6f}'.format(timer() - time))
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
            Xp = np.pad(X, ((0, 0), (0,0), (p, p), (p, p)), mode='constant')
        
        output = self.fast_fprop(Xp, out_height, out_width)
        
        if self.activation_type == "ReLU":
            output = ReLU(output)
        elif self.activation_type == "None":
            output = output # no activation
        else:
            print("error: unknown activation type")
            output = output
        return output


    def get_output_dims(self, X): # we need it because not every filter size can be applied
        batch_size, in_channels, in_height, in_width = X.shape
        assert in_channels == self.in_channels
        assert (in_height - self.height + 2*self.padding) % self.stride == 0
        assert (in_width - self.width + 2*self.padding) % self.stride == 0
        out_height = (in_height - self.height + 2*self.padding) // self.stride + 1
        out_width = (in_width - self.width + 2*self.padding) // self.stride + 1
        return out_height, out_width

    def get_im2col_ind(self, X):
        out_height, out_width = self.get_output_dims(X)
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
        return k, i, j
            
    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        """
        What shall we do here:
        1. Compute derivative of activ. func. in cur_out_batch
        2. Multiply transposed prev. weights (prev_out_batch) by error batch
        3. Multiply 1 by 2              
        """

        if self.activation_type == "ReLU":
            error_batch[cur_out_batch <= 0] = 0 # Step 1
        elif self.activation_type != "None":
            print("error: unknown activation type")

        X = prev_out_batch # previous output of the layer
        batch_size, in_channels, in_height, in_width = X.shape
        out_height, out_width = self.get_output_dims(X)
       
        k, i, j = self.get_im2col_ind(X)
        p = self.padding
        if p == 0:
            Xp = X
        else:
            Xp = np.pad(X, ((0, 0), (0,0), (p, p), (p, p)), mode='constant')
        X_col = Xp[:, k, i, j]  
 
        X_col = X_col.transpose(1, 2, 0).reshape(self.height * self.width * in_channels, -1)
        # Here we just transposed X into columns, in the same way as in forward phase
        
        # here we sum up all errors, reshape them into matrix as well
        db = np.sum(error_batch, axis=(0, 2, 3)) # 
        # db = db.reshape(self.out_channels, -1) # problems with dimensions in big network
        
        # Here - we reshape batch of errors in order to multiply it by weights
        dout_reshaped = error_batch.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        dW = np.matmul(dout_reshaped, X_col.T)
        dW = dW.reshape(self.W.shape)
        W_reshape = self.W.reshape(self.out_channels, -1)
        dX_col = np.matmul(W_reshape.T, dout_reshaped)
        
        # Reshape dX back
        dX_reshaped = dX_col.reshape(self.in_channels * self.height * self.width, -1, batch_size).transpose(2, 0, 1)
        h_pad, w_pad = in_height + 2*self.padding, in_width + 2*self.padding

        x_pad = np.zeros((batch_size, self.in_channels, h_pad, w_pad), dtype = dX_col.dtype)
        np.add.at(x_pad, (slice(None), k, i, j), dX_reshaped) # SLOW THING
        # remove padding (if any)
        if self.padding == 0:
            dX = x_pad
        else:
            dX = x_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
       
        return dW, db, dX

    def get_W_shape(self):
        return self.W.shape

    def get_b_shape(self):
        return self.b.shape

    def update(self, update_W, update_b):
        self.W -= update_W
        self.b -= update_b

    def dump_layer_info(self):
        dict_layer = {"in_channels": self.in_channels,
                      "out_channels": self.out_channels,
                      "height": self.height,
                      "width": self.width,
                      "stride": self.stride,
                      "padding": self.padding,
                      "activation_type": self.activation_type,
                      "W": self.W,
                      "b": self.b}
        layer_info = ["convlayer", dict_layer]
        return layer_info

    def get_layer_description(self):
        return 'CL [{}, {}]x{} (s{}, p{}, {})'.format(self.height, self.width, self.out_channels, self.stride, self.padding, self.activation_type)










