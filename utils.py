import numpy as np
from pathlib import Path
import pickle

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

def get_data_fast(name):
    data_csv_path = Path('.').resolve().parent/"data"/(name + ".csv")
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
