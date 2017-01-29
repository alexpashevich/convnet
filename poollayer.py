from utils import get_im2col_indices

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

    def forwardprop(self, X):
        print(X.shape)
        n, d, h, w = X.shape
        X_reshaped = X.reshape(n * d, 1, h, w) # this is necessary to apply the im2col trick, as the pool filters have depth 1
        k, i, j = get_im2col_indices(in_channels=1, height=h, width=w, out_height=self.size, out_width=self.size, stride=self.stride)
        X_col = X[:, k, i, j]  # this should be of size (size*size) x (n*d*h*w*/size/size)

        # we apply the maxpooling and save indexes in an intermediate variable in order to do the backprop after
        max_ids = np.argmax(X_col, axis=0) # this should be of size 1 x (n*d*h*w*/size/size)

        # we cache indexes for the backprop
        self.last_max_ids = max_ids

        # the desired output
        out = X_col[max_ids, range(max_ids.size)]

        # we reshape in order to get shape = (h/size) x (w/size) x n x d
        out = out.reshape(h / size, w / size, n, d)

        # and finaly shape = n x d x (h/size) x (w/size)
        out = out.transpose(2, 3, 0, 1)
        return X_col

    def backprop(self, error_batch, cur_out_batch, prev_out_batch):
        dA_col = np.zeros(cur_out_batch.shape)
        error_batch_flat = error_batch.transpose(2, 3, 0, 1).ravel()

        # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
        # Transpose step is necessary to get the correct arrangement
        # dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        # Fill the maximum index of each column with the gradient

        # Essentially putting each of the 9800 grads
        # to one of the 4 row in 9800 locations, one at each column
        dA_col[self.last_max_ids, range(self.last_max_ids.size)] = error_batch

        n, d, h, w = prev_out_batch.shape

        # We now have the stretched matrix of 4x9800, then undo it with col2im operation
        # dX would be 50x1x28x28
        dA = col2im_indices(dA_col, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)

        # Reshape back to match the input dimension: 5x10x28x28
        dA = dA.reshape(prev_out_batch.shape)

        self.last_max_ids = None
        return None, None, dA # TODO: check if dW and db == None => do not do any update
