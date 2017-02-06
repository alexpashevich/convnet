import numpy as np
from pathlib import Path
import pickle, logging
from scipy.ndimage.interpolation import rotate
log = logging.getLogger(__name__)


def ReLU(input_array):
    # rectified linear unit activation function
    return np.maximum(input_array, 0)


def sigmoid(input_array): # not used now
    # sigmoid activation function
    return 1. / (1 + np.exp(-input_array))


def vis_img(x):
    """ Take image of dims [channels, h, w], show"""
    import matplotlib.pyplot as plt
    img = x.reshape(3, 32, 32).transpose(1, 2, 0) + [0.25, 0.2, 0.2]
    print(img - [0.25, 0.2, 0.2])
    plt.imshow(img)
    plt.show()


def labels2vectors(y_labels, nb_casses):
    y_vector = np.zeros((y_labels.shape[0], nb_casses))
    y_vector[range(y_labels.shape[0]), y_labels[range(y_labels.shape[0])]] = 1
    return y_vector


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def get_data_fast(name):
    data_csv_path = Path('.').resolve()/"Data"/(name + ".csv")
    data_pkl_path = data_csv_path.parent/(name+".pkl")
    if data_pkl_path.exists():
        with data_pkl_path.open('rb') as f:
            data = pickle.load(f)
    else:
        data = np.genfromtxt(fname = str(data_csv_path), skip_header = True if name == "Ytr" else False, delimiter = ",")
        with data_pkl_path.open('wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return data


def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding ='bytes')
    fo.close()
    return dict


def prepro_mnist(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean


def prepro_cifar(X_train, X_val, X_test, img_shape):
    # for i in range(0, 3072, 1024):
    #     mean = np.mean(X_train[:,i:i+1024])
    #     std = np.std(X_train[:,i:i+1024])
    #     X_train[:,i:i+1024] = (X_train[:,i:i+1024] - mean) / std
    #     X_val[:,i:i+1024] = (X_val[:,i:i+1024] - mean) / std
    #     X_test[:,i:i+1024] = (X_test[:,i:i+1024] - mean) / std

    X_train = X_train.reshape(-1, *img_shape)
    X_val = X_val.reshape(-1, *img_shape)
    X_test = X_test.reshape(-1, *img_shape)

    return X_train, X_val, X_test


def data_augmentation(X, y, rotation_angle):
    X_flipped = np.flip(X, 3)
    X_rotated_right = rotate(X, rotation_angle, (2,3), reshape=False)
    X_rotated_left = rotate(X, -rotation_angle, (2,3), reshape=False)
    # import pudb; pudb.set_trace()
    X_aug = np.concatenate((X, X_flipped, X_rotated_right, X_rotated_left))
    y_aug = np.concatenate((y, y, y, y))
    return X_aug, y_aug


"""
im2col trick
Courtesy of :
    https://github.com/wiseodd/hipsternet/blob/f4b46e0a7856e45553955893b266df60bae8083c/hipsternet/im2col.py
"""
def get_im2col_indices(in_channels, height, width, out_height, out_width, stride):
    i0 = np.repeat(np.arange(height), width) 
    i0 = np.tile(i0, in_channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(width), height * in_channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(in_channels), height * width).reshape(-1, 1)
    k = k.astype(int)
    i = i.astype(int)
    j = j.astype(int)
    return k, i, j
