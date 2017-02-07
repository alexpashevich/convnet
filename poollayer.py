import numpy as np
from utils import unpickle
# import matplotlib.pyplot as plt


class PoolLayer(object):

    def __init__(self, layer_info):
        self.stride = layer_info["stride"]
        self.size = layer_info["size"]
        if layer_info["type"] == "maxpool":
            self.type = "maxpool" # only maxpool is implemented so far
        else:
            print("error: unknown pooling type")
        self.last_max_ids = None
        self.last_X_col = None

    def get_im2col_ind(self, out_height, out_width):
        i0 = np.repeat(np.arange(self.size), self.size) 
        i0 = np.tile(i0, 1)
        i1 = self.stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(self.size), self.size)
        j1 = self.stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(1), self.size * self.size).reshape(-1, 1)
        k = k.astype(int)
        i = i.astype(int)
        j = j.astype(int)
        return k, i, j

    def forwardprop(self, X):
         # print("X.shape = ", X.shape)
        n, d, h, w = X.shape
        # we want even height and width of the picture
        assert h % self.size == 0
        assert w % self.size == 0
        h_out = h//self.size
        w_out = w//self.size 
        X_reshaped = X.reshape(n * d, 1, h, w) # this is necessary to apply the im2col trick, as the pool filters have depth 1
        k, i, j = self.get_im2col_ind(h_out, w_out)
        X_col = X_reshaped[:, k, i, j]
        X_col = X_col.transpose(1, 2, 0).reshape(self.size*self.size, -1)
        # we apply the maxpooling and save indexes in an intermediate variable in order to do the backprop after
        max_ids = np.argmax(X_col, axis=0) # this should be of size 1 x (n*d*h*w*/size/size)
        # print("X_col.shape = ", X_col.shape)
        # we cache indexes and X_col for the backprop
        self.last_max_ids = max_ids
        self.last_X_col = X_col
        # the desired output
        out = X_col[max_ids, range(max_ids.size)]
        # we reshape in order to get shape = 
        # (h/size) x (w/size) x n x d
        out = out.reshape(h_out, w_out, n, d)
        # and finaly shape = n x d x (h/size) x (w/size)
        out = out.transpose(2, 3, 0, 1)
        # print("out.shape = ", out.shape)
        return out

    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        dA_col = np.zeros(self.last_X_col.shape)
        # transposition nd flattening to get the correct arrangement
        error_batch_flat = error_batch.transpose(2, 3, 0, 1).ravel()
        dA_col[self.last_max_ids, range(self.last_max_ids.size)] = error_batch_flat
        n, d, h, w = prev_out_batch.shape
        
        H, W = cur_out_batch.shape[2:4]
        k, i, j = self.get_im2col_ind(H, W)
        dA_col_res = dA_col.reshape(self.size*self.size, -1, n*d).transpose(2,0,1)
        A_pad = np.zeros((n*d, 1, h, w), dtype = dA_col.dtype)

        np.add.at(A_pad, (slice(None), k, i, j), dA_col_res)
        dA = A_pad.reshape(prev_out_batch.shape)
        self.last_max_ids = None
        self.last_X_col = None

        return np.ones(1), np.ones(1), dA


    def get_W_shape(self):
        return (1, 1)

    def get_b_shape(self):
        return (1, 1)

    def update(self, _1, _2):
        pass

    def assert_pool_layer(self):
        # get CIFAR_orig data
        data = unpickle("../cifar_orig/data_batch_5")
        X_train_full = data[b'data']
        y_train_full = np.array(data[b'labels'])

        # batch = np.random.randint(255, size=(2, 3, 8, 8))
        img1 = X_train_full[0,:].reshape(3, 32, 32)
        img2 = X_train_full[1,:].reshape(3, 32, 32)
        batch = np.zeros((2, 3, 32, 32))
        batch[0,:,:,:] = img1
        batch[1,:,:,:] = img2

        nb_imgs, nb_channels, height, width = batch.shape

        assert height % 2 == 0
        assert width % 2 == 0

        # we generate error to propagate
        errors = np.random.randint(255, size=(nb_imgs, nb_channels, int(height / 2), int(width / 2)))
        out_silly = np.zeros((nb_imgs, nb_channels, int(height / 2), int(width / 2)))
        out_back_silly = np.zeros(batch.shape)

        for k in range(nb_imgs):
            for ch in range(nb_channels):
                for i in range(int(height / 2)):
                    for j in range(int(height / 2)):
                        out_silly[k, ch, i, j] = np.max([batch[k, ch, 2*i, 2*j],
                                                         batch[k, ch, 2*i+1, 2*j],
                                                         batch[k, ch, 2*i, 2*j+1],
                                                         batch[k, ch, 2*i+1, 2*j+1]])
                        max_ind = np.argmax([batch[k, ch, 2*i, 2*j], batch[k, ch, 2*i+1, 2*j], batch[k, ch, 2*i, 2*j+1], batch[k, ch, 2*i+1, 2*j+1]])
                        if max_ind == 0:
                            out_back_silly[k, ch, 2*i, 2*j] = out_silly[k, ch, i, j]
                        elif max_ind == 1:
                            out_back_silly[k, ch, 2*i+1, 2*j] = out_silly[k, ch, i, j]
                        elif max_ind == 2:
                            out_back_silly[k, ch, 2*i, 2*j+1] = out_silly[k, ch, i, j]
                        else:
                            out_back_silly[k, ch, 2*i+1, 2*j+1] = out_silly[k, ch, i, j]

        out_smart = self.forwardprop(batch)
        _, _, out_back_smart = self.backprop(out_silly, out_silly, out_back_silly)

        # import pudb; pudb.set_trace()

        print("diff with the assert of maxpool (forward) is =", np.mean(out_silly - out_smart))
        print("diff with the assert of maxpool (back) is =", np.mean(out_back_silly - out_back_smart))

    def dump_layer_info(self):
        layer_info = ["poollayer", {"stride": self.stride,
                                   "size": self.size,
                                   "type": self.type}]
        return layer_info

    # def load_layer_info(self, dict_layer):
    #     self.stride = dict_layer["self.stride"]
    #     self.size = dict_layer["self.size"]
    #     self.type = dict_layer["self.type"]
    #     self.last_max_ids = None
    #     self.last_X_col = None















    
