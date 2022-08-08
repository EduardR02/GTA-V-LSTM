import os
import random
import config
import h5py
import numpy as np
from sklearn.utils import class_weight
import ast


def get_sorted_filenames(dir_list):
    # sorts data by index only
    filenames = []
    for dir_name in dir_list:
        curr_dir_filenames = [os.path.join(dir_name, filename) for filename in os.listdir(dir_name)]
        curr_dir_filenames.sort(key=lambda x: int(os.path.splitext(x)[0][len(config.data_name) + len(dir_name) + 1:]))
        filenames += curr_dir_filenames
    return filenames


def load_data(dir_list):
    print("loading training data")
    images = []
    labels = []
    filenames = get_sorted_filenames(dir_list)
    for filename in filenames:
        with h5py.File(filename, "r") as hf:
            labels += [hf.get("labels")[:]]
            images += [hf.get("images")[:]]
    print(f"Last filename was {filenames[-1]}")
    return images, labels


def load_labels_only(dir_list):
    filenames = get_sorted_filenames(dir_list)
    labels = []
    for filename in filenames:
        with h5py.File(filename, "r") as hf:
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


def generate_sample_weights_from_class_weight_dict(labels, class_weights):
    argmax_labels = np.argmax(labels, axis=-1).astype("float32")
    for i in range(argmax_labels.shape[0]):
        argmax_labels[i] = class_weights[argmax_labels[i]]
    return argmax_labels


def convert_labels_to_time_pressed(labels, images=None):
    fps_ratio = get_fps_ratio()
    new_labels = []
    index_list = [1, 3, 4, 5]
    dict_len = len(config.outputs)
    for i in range(labels.shape[0] - fps_ratio):
        new_label = np.zeros(config.output_classes)
        index = np.argmax(labels[i])
        temp = labels[i:i+fps_ratio]
        temp = np.sum(temp, axis=0)
        temp = temp[index] / fps_ratio
        if temp <= config.counts_as_tap and index in index_list:
            if index == 1: index = 0
            else: index -= 2
            new_label[index + dict_len] = 1
        else:
            new_label[index] = 1
        new_labels.append(new_label)
    new_labels = np.stack(new_labels, axis=0)
    if images is not None:
        images = images[:-fps_ratio]
    return images, new_labels


def concat_data_from_dict(filename_dict, concat=True):
    images = []
    labels = []
    for filename in filename_dict["filenames"]:
        data_x, data_y = load_file(filename)
        images += [data_x]
        labels += [data_y]
        del data_x, data_y
    if "start_index" in filename_dict:
        idx_start, idx_stop = filename_dict["start_index"], filename_dict["stop_index"]
        list_idx = 0 if filename_dict["is_test"] else -1
        images[list_idx] = images[list_idx][idx_start:idx_stop]
        labels[list_idx] = labels[list_idx][idx_start:idx_stop]
    if concat:
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
    return images, labels


def convert_labels_to_binary(labels):
    """
    :param labels: expected to be np array of shape (samples, outputs), where outputs matches the outputs dict
    categories and indices
    :return: binary labels that match the output_base dict
    (binary meaning each label is either true or false, not one hot)
    """
    reverse_output_dict = {val: [config.outputs_base[x] for x in list(key)] if key != "nothing" else [config.outputs_base[key]] for key, val in config.outputs.items()}
    new_labels = np.zeros(shape=(labels.shape[0], len(config.outputs_base)))
    argmax_labels = np.argmax(labels, axis=-1)
    for i in range(argmax_labels.shape[0]):
        for conv_idx in reverse_output_dict[argmax_labels[i]]:
            new_labels[i][conv_idx] = 1.0
    return new_labels


def divide_dataset_lstm_compatible(filenames, test_data_size, allowed_ram=config.allowed_ram_mb, incorporate_fps=True):
    """
    Don't really care about exceeding allowed ram by one file because filesizes are so small
    Needs to be rewritten if that stops to be the case
    """
    res_list = []
    filename_dict = {"filenames": [], "is_test": False}
    left_mem = allowed_ram
    full_seq_len = config.sequence_len * get_fps_ratio() if incorporate_fps else config.sequence_len
    for i in range(len(filenames) - 1, -1, -1):
        full_filename = filenames[i]
        file_size_mb = calc_filesize(full_filename)
        labels = load_file_only_labels(full_filename)
        # if length is not enough for sequencing, then skip
        if labels.shape[0] < full_seq_len:
            continue
        if test_data_size <= 0:
            filename_dict["is_test"] = False
        elif len(labels) > test_data_size:
            if test_data_size < full_seq_len or len(labels) - test_data_size < full_seq_len:
                filename_dict["is_test"] = True
                res_list.insert(0, filename_dict)
                filename_dict = {"filenames": [], "is_test": False}
            else:
                filename_dict["stop_index"] = len(labels)
                filename_dict["start_index"] = len(labels) - test_data_size
                filename_dict["is_test"] = True
                filename_dict["filenames"].insert(0, full_filename)
                res_list.insert(0, filename_dict)

                filename_dict = {"filenames": [full_filename], "stop_index":  len(labels) - test_data_size,
                                 "start_index": 0, "is_test": False}
            left_mem = allowed_ram
            test_data_size = 0
            continue
        else:
            test_data_size -= len(labels)
            filename_dict["is_test"] = True

        if file_size_mb > left_mem:
            left_mem = allowed_ram
            res_list.insert(0, filename_dict)
            filename_dict = {"filenames": [], "is_test": False}
        filename_dict["filenames"].insert(0, full_filename)
        left_mem -= file_size_mb
    if filename_dict["filenames"]:
        res_list.insert(0, filename_dict)
    return res_list


def divide_dataset(filenames, allowed_ram_mb, test_data_size=0, incorporate_fps=True):
    # because of kers utils generator, no need to account for sequence length as it will be generated on the go,
    # but files still have to be kept separate
    file_size_limit = allowed_ram_mb
    res_filenames_dict_list = []
    full_seq_len = config.sequence_len * get_fps_ratio() if incorporate_fps else config.sequence_len
    for i in range(len(filenames) - 1, -1, -1):
        full_filename = filenames[i]
        divider = {"filename": full_filename}
        labels = load_file_only_labels(full_filename)
        # if length is not enough for sequencing, then skip
        if labels.shape[0] < full_seq_len:
            continue
        # only load images if necessary
        file_size_mb = calc_filesize(full_filename)
        if test_data_size <= 0:
            divider["test_data"] = False
        elif test_data_size < len(labels):
            if test_data_size > full_seq_len:
                new_divider = div_test_data_helper(full_filename, test_data_size,
                                                   file_size_limit, is_test=True)
                res_filenames_dict_list.insert(0, new_divider)
            if len(labels) - test_data_size > full_seq_len:
                new_divider = div_test_data_helper(full_filename, test_data_size,
                                                   file_size_limit, is_test=False)
                res_filenames_dict_list.insert(0, new_divider)
            test_data_size = 0
            continue
        else:
            test_data_size -= len(labels)
            divider["test_data"] = True

        calc_chunks_and_indices(file_size_mb, file_size_limit, len(labels), divider)
        res_filenames_dict_list.insert(0, divider)
    return res_filenames_dict_list


def div_test_data_helper(filename, test_data_size, file_size_limit, is_test=True):
    m_byte = (1024 ** 2)
    new_divider = {"filename": filename, "test_data": is_test}
    images, labels = load_file(filename)
    original_len = len(labels)
    if is_test:
        images, labels = images[-test_data_size:], labels[-test_data_size:]
    else:
        images, labels = images[:-test_data_size], labels[:-test_data_size]
    file_size_mb = images.nbytes // m_byte + labels.nbytes // m_byte
    offset = original_len - len(labels) if is_test else 0
    calc_chunks_and_indices(file_size_mb, file_size_limit, len(labels), new_divider, offset)
    return new_divider


def calc_chunks_and_indices(file_size_mb, file_size_limit, data_len, divider, offset=0):
    divider["chunks"] = file_size_mb // file_size_limit + 1
    step = data_len / divider["chunks"]
    divider["indices"] = [int(round(chunk * step)) + offset for chunk in range(divider["chunks"] + 1)]


def calc_filesize(full_filename):
    m_byte = 1024 ** 2
    file_size_mb = os.stat(full_filename).st_size // m_byte
    return file_size_mb


def get_class_weights(dirs, test_data_size=0, labels=None, convert_time_pressed=False):
    if labels is None:
        labels = load_labels_only(dirs)
        # remove last x rows
        labels = np.concatenate(labels, axis=0)
    if convert_time_pressed:
        _, labels = convert_labels_to_time_pressed(labels)

    if test_data_size:
        labels = labels[:-test_data_size]
    num_classes = labels.shape[-1]
    labels = np.argmax(labels, axis=-1)
    classes = np.asarray(range(num_classes))
    inverse_proportions = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
    inverse_proportions = dict(enumerate(inverse_proportions))
    del labels
    return inverse_proportions


def get_class_weights_bin(dirs, test_data_size=0, labels=None):
    if labels is None:
        labels = load_labels_only(dirs)
        # remove last x rows
        labels = np.concatenate(labels, axis=0)
        labels = convert_labels_to_binary(labels)
    if test_data_size:
        labels = labels[:-test_data_size]
    class_sum = np.sum(labels, axis=0)
    class_weights = 1.0 / (class_sum / np.sum(class_sum))
    if np.min(class_weights) != 0:
        class_weights /= np.min(class_weights)
    # balance
    class_weights /= (np.sum(class_weights) / class_weights.shape[-1])
    class_dict = dict(enumerate(class_weights))
    del labels
    return class_dict


def convert_bin_labels_to_mse(labels, images=None):
    # question is whether to include "do nothing" or not, maybe easier if yes
    fps_ratio = get_fps_ratio()     # take avg of next x labels
    res_labels = np.zeros(labels.shape)[:-fps_ratio].astype("float32")
    for i in range(res_labels.shape[0]):
        label_sum = np.sum(labels[i:i+fps_ratio], axis=0)
        res_labels[i] = label_sum
    res_labels /= fps_ratio
    if images is not None:
        images = images[:-fps_ratio]
    return res_labels, images


def convert_any_labels_to_max_of_next(labels, next_amt):
    amt_classes = labels.shape[-1]
    for i in range(labels.shape[0]):
        labels[i] = np.sum(labels[i:i+next_amt], axis=0)
    argmax = np.argmax(labels, axis=-1).reshape(-1)
    one_hot_maxes = np.eye(amt_classes)[argmax]
    return one_hot_maxes


def test_divide_dataset(dirs):
    filenames = get_sorted_filenames(dirs)
    if config.random_file_order_train:
        random.shuffle(filenames)
    filenames = divide_dataset(filenames, config.allowed_ram_mb, 10000, incorporate_fps=False)
    for entry in filenames:
        print(entry)


def get_inverse_proportions(data):
    print(len(data))
    x = np.sum(data, axis=0)     # sum each label for each timestep separately
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    x = np.ones(x.shape) / x
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    print(x)
    print(x.shape)
    return x


def load_training_file_list(filename):
    with open(filename) as f:
        data = ast.literal_eval(f.read())
        return data


def get_fps_ratio():
    return int(round(config.fps_at_recording_time / config.fps_at_test_time))


if __name__ == "__main__":
    labels = load_labels_only([config.new_data_dir_name])
    labels = np.concatenate(labels)
    labels = convert_labels_to_binary(labels)
    labels, _ = convert_bin_labels_to_mse(labels)
    print(np.sum(labels) / labels.shape[0])
    #labels = convert_any_labels_to_max_of_next(labels, get_fps_ratio())"""
    print(labels[:100])
    class_weights = get_class_weights_bin("don't care", labels=labels)
    print(class_weights)
