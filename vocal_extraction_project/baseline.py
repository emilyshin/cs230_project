## CS230 Project
## Author: Chia-Wei Cheng
## cwcheng@stanford.edu

import numpy as np
import musdb
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tensorflow as tf
import resampy
import os
from utils import *
import keras

def baseline_model(num_epoch):
    # define the path to the training and validation set
    cur_dir = os.getcwd()
    train_x_dir = os.path.join(cur_dir, "data/train/X/")
    train_y_dir = os.path.join(cur_dir, "data/train/Y/")
    val_x_dir = os.path.join(cur_dir, "data/val/X/")
    val_y_dir = os.path.join(cur_dir, "data/val/Y/")

    # load audio data and preprocess them as a tf dataset
    batch_size = 64

    train_set = create_audio_to_audio_dataset(train_x_dir, train_y_dir)
    val_set = create_audio_to_audio_dataset(val_x_dir, val_y_dir)

    train_set = (train_set.batch(batch_size).prefetch(tf.data.AUTOTUNE))
    val_set = (val_set.batch(batch_size).prefetch(tf.data.AUTOTUNE))

    print("finished preprocessing...")

    ## construct the baseline model
    # a very simple model that avoids blowing up the memory
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32000, )),
        tf.keras.layers.Rescaling(scale=1./(2**15)),
        tf.keras.layers.Dense(32000, activation="relu"),
        tf.keras.layers.Dense(32000)
    ])

    model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    print("Model compiled...")

    print(model.summary())

    history = model.fit(train_set, batch_size=batch_size, epochs=int(num_epoch), validation_data=val_set)

    print("Done...")

    # save model weights and history
    model.save("baseline_model_0.5.keras")
    np.save("model_history_0.5.npy", history.history)

    print("model and history saved...")

    return history