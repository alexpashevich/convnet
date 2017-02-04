
import numpy as np
import math
import sys
import logging
# sys.path.append('/media/d/study/Grenoble/courses/advanced_learning_models/Competition/temp/hipsternet')
sys.path.append("../hipsternet")
import hipsternet.layer as hl
import hipsternet.input_data as input_data
from sklearn.utils import shuffle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import matplotlib.pyplot as plt

from utils import vis_img, get_data_fast, get_im2col_indices
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

    def forward_pass_hipster(self, X):
        cur_input_hipster = X
        outputs_hipster = []
        cache_hipster = []
        outputs_hipster.append(cur_input_hipster)

        for layer in self.layers:
            # print("forward_pass iter")
            if type(layer) is ConvLayer and len(cur_input_hipster.shape) < 4:
                # X -> ConvLayer, we have to resize the input
                cur_input_hipster = cur_input_hipster.reshape(-1, 3, 32, 32)

            cur_input_hipster, cache = hl.conv_forward(cur_input_hipster, layer.W, np.expand_dims(layer.b, 1), stride = layer.stride, 
                                                        padding = layer.padding)
            outputs_hipster.append(np.array(cur_input_hipster))
            cache_hipster.append(cache)

        if len(cur_input_hipster.shape) > 2:
            # we used a NN without FC layers
            cur_input_hipster = cur_input_hipster.reshape(-1, cur_input_hipster.shape[1] * cur_input_hipster.shape[2] * cur_input_hipster.shape[3])

        # the softmax layer, we subtract maximum to avoid overflow
        cur_input_hipster = softmax(cur_input_hipster)
        outputs_hipster.append(cur_input_hipster)

        return outputs_hipster, cache_hipster

    def forward_pass(self, X):
        ''' return the input data X and outputs of every layer '''
        cur_input = X
        outputs = []
        outputs.append(cur_input)
        
        for layer in self.layers:
            # print("forward_pass iter")
            if type(layer) is ConvLayer and len(cur_input.shape) < 4:
                # X -> ConvLayer, we have to resize the input
                cur_input = cur_input.reshape(-1, 3, 32, 32)

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

    def backward_pass_hipster(self, errors_batch_hipster, outputs_batch_hipster, cache_hipster):
        ''' do the backward pass and return grads for W update '''
        i = 1

        grad_W_hipster = len(self.layers) * [None]
        grad_b_hipster = len(self.layers) * [None]

        for layer in reversed(self.layers):

            if type(layer) is ConvLayer:
                type_name = "convlayer"
            else:
                type_name = "fclayer"

            cur_out_hipster = np.array(outputs_batch_hipster[-i])
            prev_out_hipster = np.array(outputs_batch_hipster[-1 - i])

            if type(layer) is ConvLayer and len(errors_batch.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the last ConvLayer, we have to resize the input
                errors_batch_hipster = errors_batch_hipster.reshape(-1, cur_out_hipster.shape[1], cur_out_hipster.shape[2], cur_out_hipster.shape[3])

            if type(layer) is ConvLayer and len(prev_out.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the first ConvLayer, we have to resize the prev_out (the image itself)
                prev_out_hipster = prev_out_hipster.reshape(-1, 3, 32, 32)

            # if type(layer) is FCLayer and len(prev_out.shape) > 3:
                # print("reshaping for FCLayer")
                # layer is the first FC after convolutional layers, we have to reshape prev_out
                # prev_out = prev_out.reshape(-1, prev_out.shape[1] * prev_out.shape[2] * prev_out.shape[3])

            # we skip the last output as it contains the final classification output
            (errors_batch_hipster, dW_hipster, db_hipster) = hl.conv_backward(errors_batch_hipster, cache_hipster[-i])

            grad_W_hipster[-i] = dW_hipster
            grad_b_hipster[-i] = db_hipster
            i += 1

            # print('backward_pass for', type_name, 'computed in {:6f}'.format(timer() - time))

        return grad_W_hipster, grad_b_hipster

    def backward_pass(self, errors_batch, outputs_batch):
        ''' do the backward pass and return grads for W update '''
        i = 1
        grad_W = len(self.layers) * [None]
        grad_b = len(self.layers) * [None]

        for layer in reversed(self.layers):

            if type(layer) is ConvLayer:
                type_name = "convlayer"
            else:
                type_name = "fclayer"
            time = timer()

            cur_out = np.array(outputs_batch[-i])
            prev_out = np.array(outputs_batch[-1 - i])

            if type(layer) is ConvLayer and len(errors_batch.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the last ConvLayer, we have to resize the input
                errors_batch = errors_batch.reshape(-1, cur_out.shape[1], cur_out.shape[2], cur_out.shape[3])

            if type(layer) is ConvLayer and len(prev_out.shape) < 4:
                # print("reshaping for ConvLayer")
                # layer is the first ConvLayer, we have to resize the prev_out (the image itself)
                prev_out = prev_out.reshape(-1, 3, 32, 32)

            if type(layer) is FCLayer and len(prev_out.shape) > 3:
                # print("reshaping for FCLayer")
                # layer is the first FC after convolutional layers, we have to reshape prev_out
                prev_out = prev_out.reshape(-1, prev_out.shape[1] * prev_out.shape[2] * prev_out.shape[3])

            # we skip the last output as it contains the final classification output
            (dW, db, errors_batch) = layer.backprop(errors_batch, cur_out, prev_out)

            # print(db_hipster.shape)

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
        # for out_our, out_hipster in zip(outputs_batch, outputs_batch_hipster):
        #     error = out_our - out_hipster
        #     # error[error < 1e-3] = 0
        #     print("np.max(error) = ", np.max(error))
        #     print('hip_to_mine_conv_mistake = {}'.format(np.mean(error)))
        # print('forward_pass computed in {:6f}'.format(timer() - time))

        # we get the errors for the whole batch at once
        errors_batch = outputs_batch[-1] - Y
        loss = CE_loss_batch(Y, outputs_batch[-1])
        # print("get_minibatch_grads loss = ", loss)

        # time = timer()
        grads_W, grads_b = self.backward_pass(np.array(errors_batch), outputs_batch[:-1])
        # for grad_W_our, grad_b_our, grad_W_hipster, grad_b_hipster in zip(grads_W, grads_b, grads_W_hipster, grads_b_hipster):
        #     error_W = grad_W_our - grad_W_hipster
        #     error_b = grad_b_our - grad_b_hipster
        #     print("np.max(error_W) = ", np.max(error_W))
        #     print('hip_to_mine_grad_W_mistake = {}'.format(np.mean(error_W)))
        #     print("np.max(error_b) = ", np.max(error_b))
        #     print('hip_to_mine_grad_b_mistake = {}'.format(np.mean(error_b)))
        # print('backward_pass computed in {:6f}'.format(timer() - time))

        return grads_W, grads_b, loss


    def fit(self, X_train, y_train, K, minibatch_size, n_iter,
            X_cv = None, y_cv = None, step_size = 0.1, epsilon = 1e-8, gamma = 0.9, use_vanila_sgd = False):
        ''' train the network and adjust the weights during n_iter iterations '''

        # do the label preprocessing first
        y_train_vector = np.zeros((y_train.shape[0], K))
        for i in range(y_train.shape[0]):
            y_train_vector[i, y_train[i]] = 1

        # we will need this for RMSprop
        E_g_W = [np.zeros(layer.get_W_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. W
        E_g_b = [np.zeros(layer.get_b_shape()) for layer in self.layers] # sum of window of square gradients w.r.t. b

        loss = math.inf
        # do fixed number of iterations
        for iter in range(n_iter):
            print("Iteration %d" % iter)
            X_train, y_train_vector = shuffle(X_train, y_train_vector)
            prev_loss = loss
            loss = 0
            proc_done = 0

            # do in minibatch fashion
            for i in range(0, X_train.shape[0], minibatch_size):
                X_minibatch = X_train[i:i + minibatch_size] # TODO: check if it is okey when X_size % minibatch_size != 0
                y_minibatch = y_train_vector[i:i + minibatch_size]

                (grads_W, grads_b, minibatch_loss) = self.get_minibatch_grads(X_minibatch, y_minibatch) # implement with the backward_pass

                loss += minibatch_loss

                # print("minibatch_loss = ", minibatch_loss)

                # update matrixes E_g_W and E_g_b used in the stepsize of RMSprop
                for j in range(self.nb_layers):
                    E_g_W[j] = gamma * E_g_W[j] + (1 - gamma) * (grads_W[j] ** 2)
                    E_g_b[j] = gamma * E_g_b[j] + (1 - gamma) * (grads_b[j] ** 2)

                # do gradient step for every layer
                for j in range(self.nb_layers):
                    if use_vanila_sgd:
                        self.layers[j].update(step_size * grads_W[j], step_size * grads_b[j])
                    else:
                        # do RMSprop step
                        self.layers[j].update(step_size / np.sqrt(E_g_W[j] + epsilon) * grads_W[j],
                                              step_size / np.sqrt(E_g_b[j] + epsilon) * grads_b[j])

                # if 1. * i / X_train.shape[0] > (proc_done + 1) * 0.1:
                    # print("%d out of %d done" % (i, X_train.shape[0]))
                    # proc_done += 1

            print("Loss = %f" % (loss / X_train.shape[0]))


            if X_cv is not None and y_cv is not None:
                y_cv_vector = np.zeros((y_cv.shape[0], K))
                for i in range(y_cv.shape[0]):
                    y_cv_vector[i, y_cv[i]] = 1
                y_cv_pred = self.predict(X_cv)
                accs = (y_cv_pred == y_cv).sum() / y_cv.size
                print("Accuracy on cross validation = %f" % accs)

            if np.absolute(loss - prev_loss) < np.sqrt(epsilon):
                print("Termination criteria is true, I stop the learning...")
                break

    def predict(self, X_test):
        ''' make prediction for all elements in X_test based on the learnt model '''
        y_test = []
        for X in X_test:
            prediction = np.argmax(self.forward_pass(X)[-1])
            y_test.append(prediction)
        return np.array(y_test)

