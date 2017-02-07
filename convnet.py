import numpy as np
import math, logging, pickle
from sklearn.utils import shuffle
# import tensorflow as tf
# import matplotlib.pyplot as plt

from utils import vis_img, get_data_fast, get_im2col_indices, print_progress_bar
from timeit import default_timer as timer
from fclayer import FCLayer
from poollayer import PoolLayer
from convlayer import ConvLayer
log = logging.getLogger(__name__)


def CE_loss(y_true, y_pred):
    # cross entropy loss (multinomial regression)
    y_pred[y_pred < 1e-7] = 1e-7 # we clip values in order to prevent log of very small value
    return -np.sum(y_true * np.log(y_pred))

def CE_loss_batch(y_true_batch, y_pred_batch):
    loss = 0
    # y_pred_batch[y_pred_batch < 1e-7] = 1e-7 # we clip values in order to prevent log of very small value
    for y_true, y_pred in zip(y_true_batch, y_pred_batch):
        loss += CE_loss(y_true, y_pred)
    return loss

def square_loss(y_true, y_pred): # not used now
    # square loss
    return 1. / 2 * np.sum((y_true - y_pred) ** 2)


def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T


class ConvNet:

    def __init__(self):
        self.layers = []
        self.nb_layers = 0

    def set_img_shape(self, img_shape):
        self.img_shape = img_shape

    def add_layer(self, layer_type, layer_info):
        ''' add a layer to the NN '''
        if layer_type == "fclayer":
            self.layers.append(FCLayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "convlayer":
            self.layers.append(ConvLayer(layer_info))
            self.nb_layers += 1
        elif layer_type == "poollayer":
            self.layers.append(PoolLayer(layer_info))
            self.nb_layers += 1
        else:
            print("error: unknown layer type")


    def forward_pass(self, X):
        ''' return the input data X and outputs of every layer '''
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        
        for layer in self.layers:
            # print("forward_pass iter")
            if type(layer) is ConvLayer and len(cur_input.shape) < 4:
                # image as 1D array -> ConvLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, *self.img_shape)

            if type(layer) is FCLayer and len(cur_input.shape) > 2:
                # ConvLayer -> FCLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, cur_input.shape[1] * cur_input.shape[2] * cur_input.shape[3])

            cur_input = layer.forwardprop(cur_input)
            outputs.append(cur_input)

        if len(cur_input.shape) > 2:
            # we used a NN without FC layers
            cur_input = cur_input.reshape(-1, cur_input.shape[1] * cur_input.shape[2] * cur_input.shape[3])

        # the softmax layer, we subtract maximum to avoid overflow
        cur_input = softmax(cur_input)
        # cur_input = np.exp(cur_input - np.max(cur_input)) / np.outer(np.exp(cur_input - np.max(cur_input)).sum(axis=1), np.ones(cur_input.shape[1]))
        outputs.append(cur_input)

        return outputs


    def backward_pass(self, errors_batch, outputs_batch):
        ''' do the backward pass and return grads for W update '''
        i = 1
        grad_W = len(self.layers) * [None]
        grad_b = len(self.layers) * [None]

        for layer in reversed(self.layers):

            if type(layer) is ConvLayer:
                type_name = "convlayer"
            elif type(layer) is PoolLayer:
                type_name = "poollayer"
            elif type(layer) is FCLayer:
                type_name = "fclayer"
            # time = timer()

            cur_out = np.array(outputs_batch[-i])
            prev_out = np.array(outputs_batch[-1 - i])

            # print("type_name", type_name)
            # print("cur_out.shape", cur_out.shape)
            # print("prev_out.shape", prev_out.shape)
            # print("errors_batch.shape", errors_batch.shape)

            if (type_name == "convlayer" or type_name == "poollayer") and len(prev_out.shape) < 4:
                # print("reshaping for {}".format(type_name))
                # layer is the first ConvLayer, we have to resize the prev_out (the image itself)
                prev_out = prev_out.reshape(-1, *self.img_shape)

            if (type_name == "convlayer" or type_name == "poollayer") and len(errors_batch.shape) < 4:
                # print("reshaping for {}".format(type_name))
                # layer is the last ConvLayer, we have to resize the input
                errors_batch = errors_batch.reshape(-1, cur_out.shape[1], cur_out.shape[2], cur_out.shape[3])

            if type(layer) is FCLayer and len(prev_out.shape) > 3:
                # print("reshaping for fclayer")
                # layer is the first FC after convolutional layers, we have to reshape prev_out
                prev_out = prev_out.reshape(-1, prev_out.shape[1] * prev_out.shape[2] * prev_out.shape[3])

            # we skip the last output as it contains the final classification output
            (dW, db, errors_batch) = layer.backprop(errors_batch, cur_out, prev_out)

            grad_W[-i] = dW # we use the returned order here in order to obtain the regular order in the end
            grad_b[-i] = db # the same here

            i += 1

            # print('backward_pass for', type_name, 'computed in {:6f}'.format(timer() - time))

        return grad_W, grad_b

    def get_minibatch_grads(self, X, Y):
        ''' return gradients with respect to W and b for a minibatch '''
        # outputs_batch = x + layers outputs => len = nb_layers + 1
        # outputs_batch[layer_i] would be array of minibatch_size outputs of layer_i
        # we do the forward_pass for the whole batch at once

        # time = timer()
        outputs_batch = self.forward_pass(X)
        # print('forward_pass computed in {:6f}'.format(timer() - time))

        # we get the errors for the whole batch at once
        errors_batch = outputs_batch[-1] - Y
        loss = CE_loss_batch(Y, outputs_batch[-1])

        # time = timer()
        grads_W, grads_b = self.backward_pass(np.array(errors_batch), outputs_batch[:-1])
        # print('backward_pass computed in {:6f}'.format(timer() - time))

        return grads_W, grads_b, loss


    def fit(self,
            X_train,
            y_train,
            K,
            minibatch_size,
            nb_epochs,
            X_cv = None,
            y_cv = None,
            optimizer = 'rmsprop',
            step_size = 0.01,
            epsilon = 1e-8,
            gamma = 0.9,
            beta1 = 0.9,
            beta2 = 0.999,
            path_for_dump = None,
            proc_of_train_to_validate = 0.1):
        '''
            train the network and adjust the weights during nb_epochs iterations
            X_train:                        data samples to train on
            y_train:                        labels to train on
            K:                              number of classes
            minibatch_size:                 size of minibatch used in training
            nb_epochs:                     number of epochs of training
            X_cv:                           data samples for validation
            y_cv:                           labels for validation
            optimizer:                      type of optimizer, currently supported 'sgd', 'rmsprop', 'adam'
            step_size:                      graident descent step size
            epsilon:                        convergence criterion, also used in rmsprop and adam to avoid zero division
            gamma:                          rmsprop parameter of sliding window of squared gradient
            beta1:                          adam parameter of sliding window of gradient
            beta2:                          adam parameter of sliding window of squared gradient
            path_for_dump:                  if the path set, a dump of the network will be made every epoche
            frac_of_train_to_validate:      fraction of X_train to be used during the accuracy estimation
        '''

        # do the label preprocessing first
        y_train_vector = np.zeros((y_train.shape[0], K))
        for i in range(y_train.shape[0]):
            y_train_vector[i, y_train[i]] = 1

        # we will need this for RMSprop and Adam in caser they are used
        m_b, m_b, v_W, v_b = [], [], [], []

        if optimizer == 'rmsprop' or optimizer == 'adam':
            v_W = [np.zeros(layer.get_W_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. W
            v_b = [np.zeros(layer.get_b_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. b

        if optimizer == 'adam':
            m_W = [np.zeros(layer.get_W_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. W
            m_b = [np.zeros(layer.get_b_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. b

        loss = math.inf
        # do fixed number of epochs
        for iter in range(nb_epochs):
            log.info("Epoch {}".format(iter))
            X_train, y_train_vector = shuffle(X_train, y_train_vector)
            prev_loss = loss
            loss = 0
            proc_done = 0
            time = timer()

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                print_progress_bar(min(i + minibatch_size, X_train.shape[0]), X_train.shape[0], 'Epoch progress:', 'Complete', length = 50)

                X_minibatch = X_train[i:i + minibatch_size]
                y_minibatch = y_train_vector[i:i + minibatch_size]

                (grads_W, grads_b, minibatch_loss) = self.get_minibatch_grads(X_minibatch, y_minibatch) # implement with the backward_pass
                loss += minibatch_loss

                # update matrixes v_W, v_b, m_W, m_b which will be used in rmsprop or adam gradient step
                if optimizer == 'rmsprop':
                    for j in range(self.nb_layers):
                        if type(self.layers[j]) is not PoolLayer:
                            v_W[j] = gamma * v_W[j] + (1 - gamma) * (grads_W[j] ** 2)
                            v_b[j] = gamma * v_b[j] + (1 - gamma) * (grads_b[j] ** 2)

                if optimizer == 'adam':
                    for j in range(self.nb_layers):
                        if type(self.layers[j]) is not PoolLayer:
                            m_W[j] = beta1 * m_W[j] + (1 - beta1) * grads_W[j]
                            m_b[j] = beta1 * m_b[j] + (1 - beta1) * grads_b[j]

                            v_W[j] = beta2 * v_W[j] + (1 - beta2) * (grads_W[j] ** 2)
                            v_b[j] = beta2 * v_b[j] + (1 - beta2) * (grads_b[j] ** 2)

                # do gradient step for every layer
                for j in range(self.nb_layers):
                    if optimizer == 'sgd':
                        if type(self.layers[j]) is not PoolLayer: # TODO: remove this kind of condition
                            self.layers[j].update(step_size * grads_W[j],
                                                  step_size * grads_b[j])
                    elif optimizer == 'rmsprop':
                        if type(self.layers[j]) is not PoolLayer:
                            self.layers[j].update(step_size / np.sqrt(v_W[j] + epsilon) * grads_W[j],
                                                  step_size / np.sqrt(v_b[j] + epsilon) * grads_b[j])
                    elif optimizer == 'adam':
                        if type(self.layers[j]) is not PoolLayer:
                            self.layers[j].update(step_size / (np.sqrt(v_W[j] / (1 - beta2)) + epsilon) * m_W[j] / (1 - beta1),
                                                  step_size / (np.sqrt(v_b[j] / (1 - beta2)) + epsilon) * m_b[j] / (1 - beta1))
                    else:
                        raise ValueError('error: unknown optimizer {}'.format(optimizer))

            log.info("Loss = {}".format(loss / X_train.shape[0]))

            if path_for_dump != None:
                self.dump_nn(path_for_dump/"{}.dump".format(iter))


            if X_cv is not None and y_cv is not None:
                y_cv_pred = self.predict(X_cv)
                accs = (y_cv_pred == y_cv).sum() / y_cv.size
                log.info("Accuracy on validation = {}".format(accs))

            if proc_of_train_to_validate > 0:
                sampled_indexes_train = np.random.choice(X_train.shape[0], int(proc_of_train_to_validate * X_train.shape[0]), replace=False)
                y_train_pred = self.predict(X_train[sampled_indexes_train])
                accs = (y_train_pred == y_train[sampled_indexes_train]).sum() / y_train[sampled_indexes_train].size
                log.info("Accuracy on train = {}".format(accs))

            epoch_time = timer() - time
            log.info('epoch is computed in {}m {}s'.format(epoch_time // 60, int((timer() - time) % 60)))

            if np.absolute(loss - prev_loss) < np.sqrt(epsilon):
                log.info("Termination criteria is true, I stop the learning...")
                break

    def predict(self, X_test):
        ''' make prediction for all elements in X_test based on the learnt model '''
        y_test = []
        for X in X_test:
            prediction = np.argmax(self.forward_pass(X)[-1])
            y_test.append(prediction)
        return np.array(y_test)

    def dump_nn(self, path_for_dump):
        layers_dumps = []
        for layer in self.layers:
            layers_dumps.append(layer.dump_layer_info())

        dict_nn = {"self.img_shape": self.img_shape,
                "self.nb_layers": self.nb_layers,
                "layers_dumps": layers_dumps}

        with path_for_dump.open(mode='wb') as file_for_dump:
            pickle.dump(dict_nn,file_for_dump)

    def load_nn(self, path_of_dump):
        with path_of_dump.open(mode='rb') as file_of_dump:
            dict_nn = pickle.load(file_of_dump)

        self.img_shape = dict_nn["self.img_shape"]
        self.nb_layers = dict_nn["self.nb_layers"]
        for layer_info in dict_nn["layers_dumps"]:
            layer_type = layer_info[0]
            if layer_type == "fclayer":
                self.layers.append(FCLayer(layer_info[1]))
            elif layer_type == "convlayer":
                self.layers.append(ConvLayer(layer_info[1]))
            elif layer_type == "poollayer":
                self.layers.append(PoolLayer(layer_info[1]))
            else:
                raise ValueError("error: can not load nn, unknown layer type {}".format(layer_type))

    def get_description(self):
        description = ''
        for layer in self.layers:
            description = description + layer.get_layer_description() + '\n'
        return description








