import os
import config
import h5py
import numpy as np


def get_sorted_filenames():
    # sorts data by index only
    filenames = os.listdir(config.new_data_dir_name)
    filenames.sort(key=lambda x: int(os.path.splitext(x)[0][len(config.data_name) + 1:]))
    return filenames


def load_data():
    print("loading training data")
    images = []
    labels = []
    filenames = get_sorted_filenames()
    for filename in filenames:
        with h5py.File(config.new_data_dir_name + filename, "r") as hf:
            labels += [hf.get("labels")[:]]
            images += [hf.get("images")[:]]
    print(f"Last filename was {filenames[-1]}")
    return images, labels


def load_labels_only():
    filenames = get_sorted_filenames()
    labels = []
    for filename in filenames:
        with h5py.File(config.new_data_dir_name + filename, "r") as hf:
            labels += [hf.get("labels")[:]]
    return labels


def load_file(filename):
    with h5py.File(filename, "r") as hf:
        images = hf.get("images")[:]
        labels = hf.get("labels")[:]
    return images, labels


def load_file_only_labels(filename):
    with h5py.File(filename, "r") as hf:
        labels = hf.get("labels")[:]
    return labels


def load_file_only_images(filename):
    with h5py.File(filename, "r") as hf:
        images = hf.get("images")[:]
    return images


def normalize_input_values(np_images, dtype):
    return np.true_divide(np_images, 127.5, dtype=dtype) - 1    # expects values between 0 and 255, returns between -1 and 1


def get_fps_ratio():
    return int(round(config.fps_at_recording_time / config.fps_at_test_time))
