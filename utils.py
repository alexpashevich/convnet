import numpy as np
from pathlib import Path
import pickle, logging
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
    plt.imshow(img)
    plt.show()


def labels2vectors(y_labels, nb_casses):
    y_vector = np.zeros((y_labels.shape[0], nb_casses))
    y_vector[range(y_labels.shape[0]), y_labels[range(y_labels.shape[0])]] = 1
    return y_vector


def train_test_split(X, y, test_size):
    indexes_test = np.random.choice(X.shape[0], int(test_size * X.shape[0]), replace=False)
    indexes_train = list(set(range(X.shape[0])) - set(indexes_test))
    X_train = X[indexes_train]
    y_train = y[indexes_train]
    X_test = X[indexes_test]
    y_test = y[indexes_test]
    return X_train, X_test, y_train, y_test, indexes_test


# def contrast_normalization(img, s=1, epsilon=1e-8, lmbd=0):
#     img_mean = np.mean(img)
#     img_var = np.mean((img - img_mean) ** 2)
#     img_norm = s*(img - img_mean) / max(epsilon, np.sqrt(lmbd + img_var))


# def zca_whitening(inputs, epsilon=1e-8):
#     sigma = np.dot(inputs, inputs.T) / inputs.shape[1] #Correlation matrix
#     U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
#     ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T) #ZCA Whitening matrix
#     return np.dot(ZCAMatrix, inputs)   #Data whitening

def shuffle(*ndarray_list):
    assert len(ndarray_list) > 0
    if type(ndarray_list[0]) == np.ndarray:
        size = ndarray_list[0].shape[0]
    else:
        size = len(ndarray_list[0])
    for ndarray in ndarray_list:
        if type(ndarray) == np.ndarray:
            cur_size = ndarray.shape[0]
        else:
            cur_size = len(ndarray)
        if cur_size != size:
            raise ValueError("error: inconsisten shapes in shuffle")

    shuffled_indexes = np.random.permutation([i for i in range(size)])
    shuffled_ndarray_list = []
    for ndarray in ndarray_list:
        shuffled_ndarray_list.append(ndarray[shuffled_indexes])

    return tuple(shuffled_ndarray_list)


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


def dump_validation_and_architecture(val_indexes_file, val_indexes, description_file, description):
    pickle.dump(val_indexes, val_indexes_file.open('wb'))
    description_file.open('w').write(description)


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
    log.info('data preprocessing...')
    for i in range(0, 3072, 1024):
        mean = np.mean(X_train[:,i:i+1024])
        std = np.std(X_train[:,i:i+1024])
        X_train[:,i:i+1024] = (X_train[:,i:i+1024] - mean) / std
        X_val[:,i:i+1024] = (X_val[:,i:i+1024] - mean) / std
        X_test[:,i:i+1024] = (X_test[:,i:i+1024] - mean) / std

    X_train = X_train.reshape(-1, *img_shape)
    X_val = X_val.reshape(-1, *img_shape)
    X_test = X_test.reshape(-1, *img_shape)

    return X_train, X_val, X_test


def data_augmentation(X, y, rotation_angle=10, prob=0.5):
    from scipy.ndimage.interpolation import rotate
    X_flipped_indexes = np.random.choice(X.shape[0], int(X.shape[0] * prob), replace=False)
    X_flipped = np.flip(X[X_flipped_indexes], 3)

    X_rotated_right_indexes = np.random.choice(X.shape[0], int(X.shape[0] * prob), replace=False)
    X_rotated_right = rotate(X[X_rotated_right_indexes], rotation_angle, (2,3), reshape=False)

    X_rotated_left_indexes = np.random.choice(X.shape[0], int(X.shape[0] * prob), replace=False)
    X_rotated_left = rotate(X[X_rotated_left_indexes], -rotation_angle, (2,3), reshape=False)

    X_aug = np.concatenate((X, X_flipped, X_rotated_right, X_rotated_left))
    y_aug = np.concatenate((y, y[X_flipped_indexes], y[X_rotated_right_indexes], y[X_rotated_left_indexes]))
    return X_aug, y_aug


def data_distortion(X, y, prob = 0.5):
    std = np.std(X)

    X_dist_bright_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
    X_dist_bright = X[X_dist_bright_ind] + std / 3
    X_dist_contrast_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
    X_dist_contrast = X[X_dist_contrast_ind]

    for i in range(0, 3072, 1024):
        mean = np.mean(X_dist_contrast[:,i:i+1024])
        X_dist_contrast[:,i:i+1024] = (X_dist_contrast[:,i:i+1024] - mean) * 1.3 + mean

    X_dist_hue_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
    X_dist_hue = X[X_dist_hue_ind]
    # for now: without fancy experiments, just changed a color a bit in one channel
    X_dist_hue[:,1024:2048] = X_dist_hue[:,1024:2048] + std / 3
    X_aug = np.concatenate((X_dist_bright, X_dist_contrast, X_dist_hue))
    y_aug = np.concatenate((y[X_dist_bright_ind], y[X_dist_contrast_ind], y[X_dist_hue_ind]))
    return X_aug, y_aug

def data_augmentation_very_new(X, y, prob = 0.5, 
        rotate=False, rotation_angle=10, 
        flip=True, 
        bright=True, bright_const = 0.02,
        contrast=True, contrast_const = 0.03,
        saturation=True, sat_const = 0.14,
        hue_bool=True, hue_const = 0.13):
    X_aug = X
    y_aug = y

    if flip==True:
        X_flip_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
        X_flipped = np.flip(X[X_flip_ind], 3)
        y_aug = np.concatenate((y_aug,y[X_flip_ind]))
        X_aug = np.concatenate((X_aug, X_flipped))

    if rotate==True:
        from scipy.ndimage.interpolation import rotate
        rotated_right_indexes = np.random.choice(X.shape[0], int(X.shape[0] * prob), replace=False)
        X_rotated_right = rotate(X[rotated_right_indexes], rotation_angle, (2,3), reshape=False)
        rotated_left_indexes = np.random.choice(X.shape[0], int(X.shape[0] * prob), replace=False)
        X_rotated_left = rotate(X[rotated_left_indexes], -rotation_angle, (2,3), reshape=False)
        X_aug = np.concatenate((X_aug, X_rotated_right, X_rotated_left))
        y_aug = np.concatenate((y_aug,y[rotated_right_indexes],y[rotated_left_indexes]))

    if bright==True:
        X_dist_bright_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
        rnd_brightness = np.random.uniform(-bright_const, bright_const)
        X_dist_bright = X[X_dist_bright_ind] + rnd_brightness
        X_aug = np.concatenate((X_aug, X_dist_bright))
        y_aug = np.concatenate((y_aug, y[X_dist_bright_ind]))

    if contrast==True:
        X_dist_contrast_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
        rnd_contrast = np.random.uniform(1-contrast_const, 1+contrast_const)
        X_dist_contrast = X[X_dist_contrast_ind]*rnd_contrast
        X_aug = np.concatenate((X_aug, X_dist_contrast))
        y_aug = np.concatenate((y_aug, y[X_dist_contrast_ind]))

    # if saturation==True:
    #     from cv2 import COLOR_HSV2RGB, COLOR_RGB2HSV, cvtColor
    #     X_dist_sat_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
    #     img_batch = X[X_dist_sat_ind]
    #     img_batch = img_batch.transpose(0,2,3,1).astype(np.float32)
    #     for i in range(len(X_dist_sat_ind)):
    #         hsv = cvtColor(img_batch[i, :, :, :], COLOR_RGB2HSV)
    #         hue, saturation, value = np.split(hsv, 3, axis=2)
    #         rnd_saturation = np.random.uniform(1-sat_const, 1+sat_const)
    #         saturation *= rnd_saturation
    #         hsv = np.dstack([hue, saturation, value])
    #         img_batch[i, :, :, :] = cvtColor(hsv, COLOR_HSV2RGB) 
    #     X_aug = np.concatenate((X_aug, img_batch.transpose(0,3,1,2).astype(np.float_)))
    #     y_aug = np.concatenate((y_aug, y[X_dist_sat_ind]))
    
    # if hue_bool==True:
    #     from cv2 import COLOR_HSV2RGB, COLOR_RGB2HSV, cvtColor
    #     X_dist_hue_ind = np.random.choice(X.shape[0], int(X.shape[0]*prob), replace=False)
    #     img_batch = X[X_dist_hue_ind]
    #     img_batch = img_batch.transpose(0,2,3,1).astype(np.float32)
    #     for i in range(len(X_dist_hue_ind)):
    #         hsv = cvtColor(img_batch[i, :, :, :], COLOR_RGB2HSV)
    #         hue, saturation, value = np.split(hsv, 3, axis=2)
    #         rnd_hue = np.random.uniform(-hue_const, hue_const)
    #         hue += rnd_hue
    #         hsv = np.dstack([hue, saturation, value])
    #         img_batch[i, :, :, :] = cvtColor(hsv, COLOR_HSV2RGB) 

    #     X_aug = np.concatenate((X_aug, img_batch.transpose(0,3,1,2).astype(np.float_)))
    #     y_aug = np.concatenate((y_aug, y[X_dist_hue_ind]))

    return X_aug, y_aug


def get_full_data(X_train_kaggle, preprocess):
    X_list = []
    y_list = []
    for i in range(1, 6):
        X_list.append(unpickle("Data/cifar_orig/data_batch_" + str(i))[b'data'])
        y_list.append(unpickle("Data/cifar_orig/data_batch_" + str(i))[b'labels'])

    X_train_full = np.concatenate((X_list[0], X_list[1], X_list[2], X_list[3], X_list[4]))
    y_train_full = np.concatenate((y_list[0], y_list[1], y_list[2], y_list[3], y_list[4]))

    if preprocess:
        for i in range(0, 3072, 1024):
            mean_full = np.mean(X_train_full[:,i:i+1024])
            std_full = np.std(X_train_full[:,i:i+1024])
            mean_kaggle = np.mean(X_train_kaggle[:,i:i+1024])
            std_kaggle = np.std(X_train_kaggle[:,i:i+1024])
            X_train_full[:,i:i+1024] = (X_train_full[:,i:i+1024] - mean_full) / std_full * std_kaggle + mean_kaggle

    return X_train_full, y_train_full


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
