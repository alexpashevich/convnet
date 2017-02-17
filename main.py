#!/usr/bin/env python3
"""
Do convnet experiments

Usage:
    main.py run kaggle_cnn
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

from our_tests import run_kaggle_cnn, predict_with_dump


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
    if args['run']:
        if args['prediction']:
            if '--dump_path' in args and args['--dump_path'] is not None:
                predict_with_dump(args['--dump_path'])
        elif args['kaggle_cnn']:
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
