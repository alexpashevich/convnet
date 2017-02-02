# from utils import get_im2col_indices
import numpy as np

# TODO: remove these 3 functions
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class PoolLayer(object):


    # TODO: add checks and asserts

    def __init__(self, layer_info):
        self.stride = layer_info["stride"]
        self.size = layer_info["size"]
        if layer_info["type"] == "maxpool":
            self.type = "maxpool" # only maxpool is implemented so far
        else:
            print("error: unknown pooling type")
        self.last_max_ids = None
        self.last_X_col = None

    def forwardprop(self, X):
        # print("X.shape = ", X.shape)
        n, d, h, w = X.shape
        X_reshaped = X.reshape(n * d, 1, h, w) # this is necessary to apply the im2col trick, as the pool filters have depth 1

        # TODO: use this function instead
        # k, i, j = get_im2col_indices_old(in_channels=1, height=self.size, width=self.size, out_height=16, out_width=16, stride=self.stride)
        # X_col = X[:, k, i, j]  # this should be of size (size*size) x (n*d*h*w*/size/size)

        X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        # print("X_col.shape = ", X_col.shape)

        # we apply the maxpooling and save indexes in an intermediate variable in order to do the backprop after
        max_ids = np.argmax(X_col, axis=0) # this should be of size 1 x (n*d*h*w*/size/size)

        # we cache indexes and X_col for the backprop
        self.last_max_ids = max_ids
        self.last_X_col = X_col

        # the desired output
        out = X_col[max_ids, range(max_ids.size)]

        # we reshape in order to get shape = (h/size) x (w/size) x n x d
        out = out.reshape(h / self.size, w / self.size, n, d)

        # and finaly shape = n x d x (h/size) x (w/size)
        out = out.transpose(2, 3, 0, 1)

        # print("out.shape = ", out.shape)

        return out

    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        dA_col = np.zeros(cur_out_batch.shape)
        error_batch_flat = error_batch.transpose(2, 3, 0, 1).ravel()

        # transposition nd flattening to get the correct arrangement
        dout_flat = error_batch.transpose(2, 3, 0, 1).ravel()

        # fill the maximum index of each column with the gradient, the rest stays zero
        dA_col[self.last_max_ids, range(self.last_max_ids.size)] = dout_flat
               
        n, d, h, w = prev_out_batch.shape

        # some more suffisticated reshaping
        dA = col2im_indices(dA_col, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)

        # reshape back to match the input dimension
        dA = dA.reshape(prev_out_batch.shape)

        self.last_max_ids = None
        self.last_X_col = None

        return None, None, dA
    
