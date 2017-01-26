import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_iris, make_moons
from sklearn.cross_validation import train_test_split

from yann import run_nn

def ReLU(input_array):
    # rectified linear unit activation function
    return np.maximum(input_array, 0)

def CE_loss(y_true, y_pred):
    # cross entropy loss (multinomial regression)
    return -np.sum(y_true * np.log(y_pred))

def sigmoid(input_array): # not used now
    # sigmoid activation function
    return 1. / (1 + np.exp(-input_array))

def square_loss(y_true, y_pred): # not used now
    # square loss
    return 1. / 2 * np.sum((y_true - y_pred) ** 2)


class fclayer:

    def __init__(self, layer_info):
        input_size = layer_info["input_size"]
        output_size = layer_info["output_size"]
        activation_type = layer_info["activation_type"]
        self.W = np.random.randn(input_size, output_size) # * 0.01 # as in the AlexNet paper
        # print(self.W)
        self.b = np.random.randn(output_size) # * 0.01
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

class ConvNet:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def add_layer(self, layer_type, layer_info):
        ''' add a layer to the NN '''
        if layer_type == "fclayer":
            self.layers.append(fclayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "convlayer":
            self.layers.append(convlayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "poollayer":
            self.layers.append(poollayer(layer_info))
            self.nb_layers += 1
        else:
            print("error: unknown layer type")

    def forward_pass(self, X):
        ''' return the input data X and outputs of every layer '''
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        for layer in self.layers:
            cur_input = layer.forwardprop(cur_input)
            outputs.append(cur_input)

        # the softmax layer
        cur_input = np.exp(cur_input) / np.exp(cur_input).sum()
        outputs.append(cur_input)

        return outputs

    def backward_pass(self, errors_batch, outputs_batch):
        ''' do the backward pass and return grads for W update '''
        i = 1
        grad_W = len(self.layers) * [None]
        grad_b = len(self.layers) * [None]
        # errors = outputs[-1] - y_true # we expect CE loss and softmax in the end
        # print("errors = ", errors)
        for layer in reversed(self.layers):
            # we skip the last output as it contains the final classification output
            (dW, db, errors_batch) = layer.backprop(errors_batch, np.array(outputs_batch[-i]), np.array(outputs_batch[-1 - i]))
            grad_W[-i] = dW # we use the returned order here in order to obtain the regular order in the end
            grad_b[-i] = db # the same here
            i += 1

        return grad_W, grad_b

    def get_minibatch_grads(self, X, Y):
        ''' return gradients with respect to W and b for a minibatch '''
        loss = 0
        errors_batch = []
        # outputs = x + layers outputs => len = nb_layers + 1
        # outputs_batch[layer_i] would be array of minibatch_size outputs of layer_i
        outputs_batch = [[] for _ in range((self.nb_layers + 1))]

        # we do the forward_pass and stack all the outputs in the right order
        for x, y in zip(X, Y):
            outputs = self.forward_pass(x)

            errors = outputs[-1] - y
            # TODO: redo everything with fancy numpy functions instead of ugly loops
            for i in range(self.nb_layers + 1):
                outputs_batch[i].append(outputs[i])
            errors_batch.append(errors)
            loss += CE_loss(y, outputs[-1])

        grads_W, grads_b = self.backward_pass(np.array(errors_batch), outputs_batch)

        return grads_W, grads_b, loss


    def fit(self, X_train, Y_train, K, step_size, minibatch_size, n_iter):
        ''' train the network and adjust the weights during n_iter iterations '''

        # do the label preprocessing first
        Y_train_vector = np.zeros((Y_train.shape[0], K))
        for i in range(Y_train.shape[0]):
            Y_train_vector[i, Y_train[i]] = 1

        # do fixed number of iterations
        for iter in range(n_iter):
            print("Iteration %d" % iter)
            X_train, Y_train_vector = shuffle(X_train, Y_train_vector)
            loss = 0

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                X_minibatch = X_train[i:i + minibatch_size] # TODO: check if it is okey when X_size % minibatch_size != 0
                Y_minibatch = Y_train_vector[i:i + minibatch_size]

                (grads_W, grads_b, minibatch_loss) = self.get_minibatch_grads(X_minibatch, Y_minibatch) # implement with the backward_pass
                loss += minibatch_loss
                # do gradient step for every layer
                # so far the step size is fixed, smth like RMSprop should be used ideally
                for i in range(self.nb_layers):
                    self.layers[i].update(step_size * grads_W[i], step_size * grads_b[i])

            print("Loss = %f" % (loss / X_train.shape[0]))

    def predict(self, X_test):
        ''' make prediction for all elements in X_test based on the learnt model '''
        Y_test = []
        for X in X_test:
            prediction = np.argmax(self.forward_pass(X)[-1])
            Y_test.append(prediction)
        return Y_test

if __name__ == "__main__":
    X, Y = make_moons(n_samples=5000, random_state=42, noise=0.1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    # run_nn(X_train, X_test, Y_train, Y_test)

    np.random.seed(228)
    size1 = 2
    size2 = 100
    size3 = 40
    size4 = 2

    cnn = ConvNet()
    cnn.add_layer("fclayer", layer_info = {"input_size": size1, "output_size": size2, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size2, "output_size": size3, "activation_type": "ReLU"})
    cnn.add_layer("fclayer", layer_info = {"input_size": size3, "output_size": size4, "activation_type": "None"})
    cnn.fit(X, Y, K = 2, step_size = 1e-4, minibatch_size = 50, n_iter = 30)


    y_pred = cnn.predict(X_test)
    accs = (y_pred == Y_test).sum() / Y_test.size
    print('Mean accuracy: %f' % accs)












