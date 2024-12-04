## CS230 Project
## Author: Chia-Wei Cheng
## cwcheng@stanford.edu

import numpy as np
import musdb
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from conv_blstm import conv_blstm_model
from preprocessing import *
from baseline import *

def main():
    # tf.debugging.set_log_device_placement(True)
    tf.config.list_physical_devices('GPU')

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_data", dest="generate_data")
    parser.add_argument("--overlapping_rate", dest="overlapping_rate")
    parser.add_argument("--train_size", dest="train_size")
    parser.add_argument("--val_size", dest="val_size")
    parser.add_argument("--test_size", dest="test_size")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--epoch", dest="epoch")
    parser.add_argument("--batch_size", dest="batch_size")
    args = parser.parse_args()

    # data preprocessing
    with tf.device('/GPU:0'):
        if args.generate_data.lower() == "1-second_non_overlapping":
            generate_one_second_non_overlapping_samples()
        elif args.generate_data.lower() == "1-second_overlapping":
            generate_one_second_overlapping_samples(float(args.overlapping_rate), int(args.train_size), int(args.val_size), int(args.test_size))

        # run model
        if args.model == "baseline":
            baseline_model(args.epoch)
        elif args.model == "conv_blstm":
            conv_blstm_model(int(args.batch_size), int(args.epoch))


    return 0


if __name__ == "__main__":
    main()