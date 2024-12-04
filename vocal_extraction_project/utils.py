## CS230 Project
## Author: Chia-Wei Cheng
## cwcheng@stanford.edu

import numpy as np
import tensorflow as tf
import os

def load_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # Remove the channel dimension
    return audio

def load_audio_pair(input_path, target_path):
    input_audio = load_audio(input_path)
    target_audio = load_audio(target_path)
    return input_audio, target_audio

def create_audio_to_audio_dataset(input_dir, target_dir):
    """
    This function creates an audio-to-audio dataset based on the two provided directories.
    It combines each training and its corresponding target into pairs that can be processed
    by tensorflow

    :param input_dir: file path to the input directory
    :param target_dir: file path to the target directory
    :param sample_rate: fs
    :return: a tensorflow dataset
    """

    input_files = sorted([input_dir + f for f in os.listdir(input_dir) if f.endswith('.wav')])
    target_files = sorted([target_dir + f for f in os.listdir(target_dir) if f.endswith('.wav')])

    dataset = tf.data.Dataset.from_tensor_slices((input_files, target_files))

    dataset = dataset.map(lambda x, y: load_audio_pair(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


class AudioSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, source_dir, target_dir, batch_size, sample_rate=32000):
        self.source_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir)])
        self.target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
        self.batch_size = batch_size
        self.sample_rate = sample_rate

        # Ensure source and target files match
        # assert len(self.source_files) == len(self.target_files), "Mismatch in source and target files."
        self.num_samples = len(self.source_files)

    def __len__(self):
        # Number of batches
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        # Get batch file paths
        batch_source_files = self.source_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_target_files = self.target_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load and preprocess batch
        source_batch = np.array([self._load_and_preprocess_audio(file) for file in batch_source_files])
        target_batch = np.array([self._load_and_preprocess_audio(file) for file in batch_target_files])

        return source_batch, target_batch

    def _load_and_preprocess_audio(self, file_path):
        # Load audio file
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=32000)
        audio = tf.squeeze(audio, axis=-1)  # Remove channel dimension

        return audio.numpy()  # Convert Tensor to Numpy for batching