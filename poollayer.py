from utils import vis_img, get_data_fast, get_im2col_indices

class poollayer:


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
        return out

    def backprop(self, output):
        dA = np.zeros()


        self.last_max_ids = None
