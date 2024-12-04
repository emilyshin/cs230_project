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


def conv_blstm_model(batch_size, num_epoch):
    # define the path to the training and validation set
    cur_dir = os.getcwd()
    train_x_dir = os.path.join(cur_dir, "data/train/X/")
    train_y_dir = os.path.join(cur_dir, "data/train/Y/")
    val_x_dir = os.path.join(cur_dir, "data/val/X/")
    val_y_dir = os.path.join(cur_dir, "data/val/Y/")


    # load audio data and preprocess them as a tf dataset
    # batch_size = 64

    train_set = create_audio_to_audio_dataset(train_x_dir, train_y_dir)
    val_set = create_audio_to_audio_dataset(val_x_dir, val_y_dir)

    train_set = (train_set.batch(batch_size).prefetch(tf.data.AUTOTUNE))
    val_set = (val_set.batch(batch_size).prefetch(tf.data.AUTOTUNE))

    # we tried using the generator to avoid loading the entire dataset into memory
    train_generator = AudioSequenceGenerator(train_x_dir, train_y_dir, batch_size=batch_size)
    val_generator = AudioSequenceGenerator(val_x_dir, val_y_dir, batch_size=batch_size)

    print("finished preprocessing...")

    # construct the conv_blstm model
    inputs = tf.keras.Input(shape=(32000, 1))
    x = tf.keras.layers.Rescaling(scale=1. / (2 ** 15))(inputs)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=1500, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=640, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=640, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Conv1D(filters=1, padding="same", kernel_size=20)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="mae", metrics=["mae"])

    print("Model compiled...")

    print(model.summary())

    history = model.fit(train_generator, batch_size=batch_size, epochs=int(num_epoch), validation_data=val_generator)

    print("Done...")

    # save model weights and history
    model.save("conv_blstm_model.keras")
    np.save("conv_blstm_model_history.npy", history.history)

    print("model and history saved...")
    return