## CS230 Project
## Author: Chia-Wei Cheng
## cwcheng@stanford.edu

import numpy as np
import musdb
from scipy.io import wavfile
import os
import resampy


def generate_one_second_non_overlapping_samples():

    """
    This function will load audio from the musdb18 dataset and generate one-second non-overlapping audio clips
    :return: None
    """

    ##########################################################
    # load music tracks from the dataset
    ##########################################################

    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "musdb18_samples")

    mus_samples = musdb.DB(root=data_dir, subsets="samples")
    num_samples = len(mus_samples)

    print("Finished loading {} songs...".format(num_samples))

    ##########################################################
    # generate audio data into train
    ##########################################################

    # to avoid blowing up AWS memory and storage we down sample the audio to 32k Hz and use the 16-bit format
    # the raw data have been normalized to have a range [-1, 1]
    # we will use the 16-bit integer format to store the audio

    fs = 44100  # sampling rate for the dataset is 44.1k Hz
    new_fs = 32000
    window_len = new_fs
    num_bits = 16  # musdb18 dataset encodes the audio as 44.1k Hz 16-bit

    # create random access pattern to shuffle the samples
    np.random.seed(42)
    random_indices = list(range(num_samples))
    np.random.shuffle(random_indices)

    # in this baseline design we will use roughly 2 hrs (7200 secs) of songs for validation
    # we also want to use complete songs in each set to avoid having a fraction of a song is in the train set and
    # another fraction in the validation set.
    # so we have this cutoff index to indicate where to split train and validation while satisfying our goals.
    # goals: at least 2 hours and they are complete songs
    window_count = 0
    val_set_size = 7200 # in seconds

    sample_count = 1

    # to keep track of the progress of the loop below
    iteration_count = 1

    for i in random_indices:
        one_sample = mus_samples[i]
        song_name = one_sample.title
        print("processing {}".format(song_name))

        mixture_l = np.array(resampy.resample(one_sample.audio[:, 0], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16)
        mixture_r = np.array(resampy.resample(one_sample.audio[:, 1], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16)
        vocal_l = np.array(resampy.resample(one_sample.stems[4, :, 0], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16) # 4th index is the vocal component
        vocal_r = np.array(resampy.resample(one_sample.stems[4, :, 1], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16)

        # from both channels
        num_windows = len(mixture_l) // window_len

        dest_path = os.path.join(cur_dir, "data/train/")

        if window_count < val_set_size:
            dest_path = os.path.join(cur_dir, "data/val/")
            window_count += 2 * num_windows # left and right channels in each song

        for j in range(num_windows):
            # excluding the tail
            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_l[j*window_len:(j+1)*window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_l[j*window_len:(j+1)*window_len])
            sample_count += 1

            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_r[j*window_len:(j+1)*window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_r[j*window_len:(j+1)*window_len])
            sample_count += 1

        print("{}/{} songs processed...".format(iteration_count, num_samples))
        iteration_count += 1

    print("done...")

    return None


def generate_one_second_overlapping_samples(overlap_rate, train_set_size, val_set_size, test_set_size):
    ##########################################################
    # load music tracks from the dataset
    ##########################################################

    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "musdb18_samples")

    mus_samples = musdb.DB(root=data_dir, subsets="samples")
    num_samples = len(mus_samples)

    print("Finished loading {} songs...".format(num_samples))

    ##########################################################
    # generate audio data into train
    ##########################################################

    # to avoid blowing up AWS memory and storage we down sample the audio to 32k Hz and use the 16-bit format
    # the raw data have been normalized to have a range [-1, 1]
    # we will use the 16-bit integer format to store the audio

    fs = 44100  # sampling rate for the dataset is 44.1k Hz
    new_fs = 32000
    window_len = new_fs
    num_bits = 16  # musdb18 dataset encodes the audio as 44.1k Hz 16-bit

    # create random access pattern to shuffle the samples
    np.random.seed(42)
    random_indices = list(range(num_samples))
    np.random.shuffle(random_indices)

    window_count = 0
    val_set_size = val_set_size # in number of samples
    test_set_size = test_set_size
    train_set_size = train_set_size
    hop_size = int((1-overlap_rate) * new_fs) # in number of samples

    sample_count = 1

    # to keep track of the progress of the loop below
    iteration_count = 1

    # process training data
    for i in random_indices[0:train_set_size]:
        one_sample = mus_samples[i]
        song_name = one_sample.title
        print("processing {}".format(song_name))

        mixture_l = np.array(resampy.resample(one_sample.audio[:, 0], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16)
        mixture_r = np.array(resampy.resample(one_sample.audio[:, 1], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16)
        vocal_l = np.array(resampy.resample(one_sample.stems[4, :, 0], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16) # 4th index is the vocal component
        vocal_r = np.array(resampy.resample(one_sample.stems[4, :, 1], fs, new_fs)*(2**(num_bits-1)), dtype=np.int16)

        start_index = 0
        dest_path = os.path.join(cur_dir, "data/train/")

        while start_index + window_len < len(mixture_l):
            # excluding the tail
            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_l[start_index:start_index + window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_l[start_index:start_index + window_len])
            sample_count += 1

            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_r[start_index:start_index + window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_r[start_index:start_index + window_len])
            sample_count += 1

            start_index += hop_size

        print("{}/{} songs processed...".format(iteration_count, num_samples))
        iteration_count += 1

    print("finished generating training samples")

    # process validation data
    for i in random_indices[train_set_size:train_set_size+val_set_size]:
        one_sample = mus_samples[i]
        song_name = one_sample.title
        print("processing {}".format(song_name))

        mixture_l = np.array(resampy.resample(one_sample.audio[:, 0], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)
        mixture_r = np.array(resampy.resample(one_sample.audio[:, 1], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)
        vocal_l = np.array(resampy.resample(one_sample.stems[4, :, 0], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)  # 4th index is the vocal component
        vocal_r = np.array(resampy.resample(one_sample.stems[4, :, 1], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)

        start_index = 0
        dest_path = os.path.join(cur_dir, "data/val/")

        while start_index + window_len < len(mixture_l):
            # excluding the tail
            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_l[start_index:start_index + window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_l[start_index:start_index + window_len])
            sample_count += 1

            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_r[start_index:start_index + window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_r[start_index:start_index + window_len])
            sample_count += 1

            start_index += hop_size

        print("{}/{} songs processed...".format(iteration_count, num_samples))
        iteration_count += 1

    print("finished generating validation samples")

    # process test data
    for i in random_indices[train_set_size+val_set_size:train_set_size + val_set_size + test_set_size]:
        one_sample = mus_samples[i]
        song_name = one_sample.title
        print("processing {}".format(song_name))

        mixture_l = np.array(resampy.resample(one_sample.audio[:, 0], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)
        mixture_r = np.array(resampy.resample(one_sample.audio[:, 1], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)
        vocal_l = np.array(resampy.resample(one_sample.stems[4, :, 0], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)  # 4th index is the vocal component
        vocal_r = np.array(resampy.resample(one_sample.stems[4, :, 1], fs, new_fs) * (2 ** (num_bits - 1)), dtype=np.int16)

        start_index = 0
        dest_path = os.path.join(cur_dir, "data/test/")

        while start_index + window_len < len(mixture_l):
            # excluding the tail
            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_l[start_index:start_index + window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_l[start_index:start_index + window_len])
            sample_count += 1

            wavfile.write(dest_path + "X/sample{}-{}.wav".format(sample_count, song_name), new_fs, mixture_r[start_index:start_index + window_len])
            wavfile.write(dest_path + "Y/sample{}-{}.wav".format(sample_count, song_name), new_fs, vocal_r[start_index:start_index + window_len])
            sample_count += 1

            start_index += hop_size

        print("{}/{} songs processed...".format(iteration_count, num_samples))
        iteration_count += 1

    print("finished generating test samples")

    print("done...")

