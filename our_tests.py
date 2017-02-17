# import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from convnet import ConvNet
from utils import get_data_fast, get_im2col_indices, prepro_mnist, prepro_cifar, data_augmentation, data_distortion
from utils import train_test_split, dump_validation_and_architecture, vis_img, get_full_data
import pickle, csv, logging

DUMPFOLDER = Path('dumps')
log = logging.getLogger(__name__)


def run_kaggle_cnn(datetime_string, cnn_load_path = None, val_ind_path = None):
    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1].astype(int)


    if val_ind_path is None:
        X_train, X_val, y_train, y_val, val_indexes = train_test_split(X_train_full, y_train_full, test_size = 0.2)
    else:
        val_indexes = pickle.load(Path(val_ind_path).open('rb'))
        X_val, y_val = X_train_full[val_indexes], y_train_full[val_indexes]
        X_train = X_train_full[list(set(range(X_train_full.shape[0])) - set(val_indexes))]
        y_train = y_train_full[list(set(range(X_train_full.shape[0])) - set(val_indexes))]

    nb_samples, data_length, nb_classes = X_train.shape[0], X_train.shape[1], y_train.max() + 1
    img_shape = (3, 32, 32)
    
    from utils import data_augmentation_very_new
    X_train, X_val, X_test = prepro_cifar(X_train, X_val, X_test, img_shape)
    X_train, y_train = data_augmentation_very_new(X_train, y_train, prob=1, rotate=False, hue_bool=False, contrast = False,  saturation=False)

    log.info("X_train.shape = {}, X_val.shape = {}, X_test.shape = {}".format(X_train.shape, X_val.shape, X_test.shape))

    dump_folder = DUMPFOLDER/datetime_string
    dump_folder.mkdir()

    ch1 = 48
    ch2 = 64
    ch3 = 96
    ch4 = 128
    nb_classes = 10

    fc_size_in1 = 8*8*ch2
    fc_size_in2 = 1024

    cnn = ConvNet()

    if cnn_load_path is None:
        log.info('Building CNN architecture from scratch')
        cnn.set_img_shape(img_shape)
        cnn.add_layer("convlayer", layer_info = {"in_channels": img_shape[0],
                                                 "out_channels": ch1,
                                                 "height": 3,
                                                 "width": 3,
                                                 "stride": 1,
                                                 "padding": 1,
                                                 "activation_type": "ReLU"}) # 32 x 32 x ch1
        cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"}) # 16 x 16 x ch1
        cnn.add_layer("convlayer", layer_info = {"in_channels": ch1,
                                                 "out_channels": ch2,
                                                 "height": 3,
                                                 "width": 3,
                                                 "stride": 1,
                                                 "padding": 1,
                                                "activation_type": "ReLU"}) # 16 x 16 x ch2
        cnn.add_layer("poollayer", layer_info = {"stride": 2, "size": 2, "type": "maxpool"}) # 8 x 8 x ch2
        cnn.add_layer("convlayer", layer_info = {"in_channels": ch2,
                                                 "out_channels": ch3,
                                                 "height": 3,
                                                 "width": 3,
                                                 "stride": 1,
                                                 "padding": 0,
                                                 "activation_type": "ReLU"}) # 6 x 6 x ch3
        cnn.add_layer("poollayer", layer_info = {"stride":2, "size": 2, "type": "maxpool"}) # 3 x 3 x ch3
        cnn.add_layer("convlayer", layer_info = {"in_channels": ch3,
                                                 "out_channels": ch4,
                                                 "height": 3,
                                                 "width": 3,
                                                 "stride": 1,
                                                 "padding": 0,
                                                 "activation_type": "ReLU"}) # 1 x 1 x ch4
        cnn.add_layer("convlayer", layer_info = {"in_channels": ch4,
                                                 "out_channels": nb_classes,
                                                 "height": 1,
                                                 "width": 1,
                                                 "stride": 1,
                                                 "padding": 0,
                                                 "activation_type": "None"}) # 1 x 1 x 10
    else:
        log.info('Loading CNN architecture from {}'.format(cnn_load_path))
        cnn.load_nn(Path(cnn_load_path))

    dump_validation_and_architecture(dump_folder/'validation_indexes.dat', val_indexes, dump_folder/'info.txt', cnn.get_description())
  

    cnn.fit(X_train,
            y_train,
            K = nb_classes,
            X_cv = X_val,
            y_cv = y_val,
            minibatch_size = 50,
            nb_epochs = 40,
            step_size = 0.001,
            optimizer='rmsprop',
            path_for_dump = dump_folder)

    y_test = cnn.predict(X_test)
    with (dump_folder/"Yte.csv").open('w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Prediction'])
        writer.writeheader()
        for id, y in zip(range(1, y_test.shape[0] + 1), y_test):
            writer.writerow({'Id': id, 'Prediction': y})
    log.info("Prediction was done successfully!")


def predict_with_dump(dump_path):
    X_train_full = get_data_fast("Xtr")[:,:-1]
    X_test = get_data_fast("Xte")[:,:-1]
    y_train_full = get_data_fast("Ytr")[:,1].astype(int)

    X_train, X_val, y_train, y_val, _ = train_test_split(X_train_full, y_train_full, test_size = 0.01)
    img_shape = (3, 32, 32)
    X_train, X_val, X_test = prepro_cifar(X_train, X_val, X_test, img_shape)

    cnn = ConvNet()
    cnn.load_nn(Path(dump_path))
    y_test = cnn.predict(X_test)
    dump_folder = DUMPFOLDER/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dump_folder.mkdir()
    with (dump_folder/"Yte.csv").open('w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Prediction'])
        writer.writeheader()
        for id, y in zip(range(1, y_test.shape[0] + 1), y_test):
            writer.writerow({'Id': id, 'Prediction': y})
    log.info("Prediction was done successfully!")









