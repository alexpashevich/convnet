#!/usr/bin/env python3
"""
Do convnet experiments

Usage:
    main.py test (moons|kaggle|poollayer|mnist)
    main.py test conv [--valsize <float>] [--seed <int>] [options]
    main.py run kaggle_cnn
    main.py test tensorflow (full|slim)
    main.py continue kaggle_cnn [--cnn_path <string>] [--val_ind_path <string>] [options]
    main.py run prediction [--dump_path <string>] [options]

Options:
    Convolution test:
        --valsize <float>       Fraction of validation data [default: 0.1]
        --seed <int>            Seed value

    Continue kaggle_cnn:
        --cnn_path <string>     Path to the dump file with CNN parameters
        --val_ind_path <string> Path to the validation set indexes file

    Prediction run:
        --dump_path <string>    Path to the dump file with CNN parameters

    Other options:
        --logfile <str>         Log to the selected file (by default logfiles correspond to time)
"""
from pathlib import Path
from docopt import docopt
from datetime import datetime
import logging
import sys
sys.path.append("external/hipsternet")

from our_tests import test_moons, test_kaggle_fcnn, test_poollayer, run_kaggle_cnn, test_conv_layer, test_mnist, predict_with_dump
from tf_tests import test_tensorflow_full, test_tensorflow_slim


LOG_FORMATTER = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s",
                            "%Y-%m-%d %H:%M:%S")
LOGFOLDER = Path('logs')
DATETIME_NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def logging_setup():
    """ Define logging to STDOUT """
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def start_logging_to_file(filename=None):
    """ If no name - generate name automatically """
    if filename is None:
        logfilename = LOGFOLDER/(DATETIME_NOW + ".log")
    else:
        logfilename = LOGFOLDER/filename
    out_filehandler = logging.FileHandler(str(logfilename))
    out_filehandler.setFormatter(LOG_FORMATTER)
    log.addHandler(out_filehandler)
    log.info('Starting logging to file {}'.format(logfilename))

def main(args):
    if args['test']:
        if args['moons']:
            test_moons()
        elif args['kaggle']:
            test_kaggle_fcnn()
        elif args['mnist']:
            test_mnist()
        elif args['poollayer']:
            test_poollayer()
        elif args['conv']:
            valsize = float(args['--valsize'])
            seed = None
            if '--seed' in args and args['--seed'] is not None:
                seed = int(args['--seed'])
            test_conv_layer(valsize, seed)
        elif args['tensorflow']:
            if args['full']:
                test_tensorflow_full()
            elif args['slim']:
                test_tensorflow_slim()
    elif args['run']:
        if args['prediction']:
            if '--dump_path' in args and args['--dump_path'] is not None:
                predict_with_dump(args['--dump_path'])
        elif args['kaggle_cnn']:
            # run_info = {}
            # if args['--augmentation']
            run_kaggle_cnn(DATETIME_NOW)
    elif args['continue']:
        if args['kaggle_cnn']:
            if '--cnn_path' in args and '--val_ind_path' in args and args['--cnn_path'] is not None and args['--val_ind_path'] is not None:
                run_kaggle_cnn(DATETIME_NOW, cnn_load_path=args['--cnn_path'], val_ind_path=args['--val_ind_path'])

if __name__ == "__main__":
    args = docopt(__doc__)
    log = logging_setup()
    log.setLevel(logging.DEBUG)
    if '--logfile' in args:
        logfilename = args['--logfile']
    else:
        logfilename = None
    start_logging_to_file(logfilename)
    main(args)
