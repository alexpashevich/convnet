class fclayer:

    def __init__(self, layer_info):
        input_size = layer_info["input_size"]
        output_size = layer_info["output_size"]
        activation_type = layer_info["activation_type"]
        self.W = np.random.randn(input_size, output_size) * 0.01 # as in the AlexNet paper
        self.b = np.random.randn(output_size) * 0.01
        self.activation_type = activation_type # so far only ReLU is implemented

    def get_W_shape(self):
        return self.W.shape

    def get_b_shape(self):
        return self.b.shape

    def forwardprop(self, X):
        out = X @ self.W + self.b

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