import numpy as np
import tensorflow as tf
from lenet import create_neural_net, create_cnn_only, replace_cnn_dense_layer, freeze_part_of_inception
from tensorflow import compat
import time
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import shutil
import datetime
import config
import utils
import random
import gc

current_data_dir = config.new_data_dir_name


def setup_tf():
    tf.keras.backend.clear_session()
    config_var = compat.v1.ConfigProto()
    config_var.gpu_options.allow_growth = True
    compat.v1.Session(config=config_var)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def train_model(load_saved, freeze=False, load_saved_cnn=False):
    setup_tf()
    if load_saved:
        model = load_model(config.model_name)
    else:
        model = create_neural_net(load_pretrained_cnn=load_saved_cnn, model_name=config.cnn_only_name)
    # freeze convolutional model to fine tune lstm (the cnn is treated as one layer
    # make sure you freeze the correct one)
    # goes the other way around too if the model was saved frozen and you want to unfreeze
    model.layers[1].trainable = not freeze
    optimizer = Adam(learning_rate=config.lr)
    # recompile to make the changes
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    custom_training_loop(model, config.allowed_ram_mb, 10000, save_every_epoch=True,
                         normalize_input_values=False, incorporate_fps=True)


def train_cnn_only(load_saved, swap_output_layer=False, freeze_part=True):
    setup_tf()
    if load_saved:
        model = load_model(config.cnn_only_name)
        if swap_output_layer:
            model = replace_cnn_dense_layer(model)
    else:
        model = create_cnn_only()
    if freeze_part:
        model = freeze_part_of_inception(model, "mixed9")
    model.summary()
    cnn_only_training(model, False)


def cnn_only_training(model, normalize_inputs=True):
    test_data_size = 10000
    class_weights = get_class_weights(test_data_size=test_data_size)
    filenames = utils.get_sorted_filenames(current_data_dir)
    random.shuffle(filenames)
    filename_dict_list = divide_dataset_cnn_only(filenames, test_data_size, normalize_inputs,
                                                 config.known_normalize_growth, allowed_ram=config.allowed_ram_mb)
    for epoch in range(config.epochs):
        for filename_dict in filename_dict_list:
            K.clear_session()
            images = []
            labels = []
            for filename in filename_dict["filenames"]:
                data_x, data_y = utils.load_file(filename)
                if normalize_inputs:
                    data_x = utils.normalize_input_values(data_x, "float32")
                images += [data_x]
                labels += [data_y]
                del data_x, data_y
            if "start_index" in filename_dict:
                idx_start, idx_stop = filename_dict["start_index"], filename_dict["stop_index"]
                list_idx = 0 if filename_dict["is_test"] else -1
                images[list_idx] = images[list_idx][idx_start:idx_stop]
                labels[list_idx] = labels[list_idx][idx_start:idx_stop]
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            images, labels = convert_labels_to_time_pressed(images, labels)
            if not filename_dict["is_test"]:
                model.fit(images, labels,
                          epochs=epoch+1, initial_epoch=epoch, batch_size=config.CNN_ONLY_BATCH_SIZE,
                          class_weight=class_weights, validation_split=0.05, shuffle=True)
            else:
                model.evaluate(images, labels, batch_size=config.CNN_ONLY_BATCH_SIZE)
            del images, labels
            gc.collect()
    model.save(config.cnn_only_name)


def divide_dataset_cnn_only(filenames, test_data_size, normalize_inputs, normalize_factor=0, allowed_ram=config.allowed_ram_mb):
    res_list = []
    filename_dict = {"filenames": [], "is_test": False}
    left_mem = allowed_ram
    for i in range(len(filenames) - 1, -1, -1):
        full_filename = current_data_dir + filenames[i]
        file_size_mb = calc_filesize(full_filename, normalize_inputs, normalize_factor)
        labels = utils.load_file_only_labels(full_filename)
        if test_data_size <= 0:
            filename_dict["is_test"] = False
        elif len(labels) > test_data_size:
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


def custom_training_loop(model, allowed_ram_mb, test_data_size, save_every_epoch, normalize_input_values=True, incorporate_fps = True):
    """log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)"""
    class_weights = get_class_weights(test_data_size=test_data_size)
    filenames = utils.get_sorted_filenames(current_data_dir)
    if config.random_file_order_train:
        random.shuffle(filenames)
    filenames = divide_dataset(filenames, allowed_ram_mb, test_data_size, normalize_input_values,
                               incorporate_fps=incorporate_fps, known_normalize_growth=config.known_normalize_growth)
    for epoch in range(config.epochs):
        for i in range(len(filenames)):
            K.clear_session()
            if os.path.isfile(filenames[i]["filename"]):
                for j in range(filenames[i]["chunks"]):
                    K.clear_session()
                    images, labels = utils.load_file(filenames[i]["filename"])
                    start_idx, stop_idx = filenames[i]["indices"][j], filenames[i]["indices"][j+1]
                    # stop_idx is next start index, therefore not stop_idx-1 because is it the first NOT included index
                    images, labels = images[start_idx:stop_idx], labels[start_idx:stop_idx]
                    if normalize_input_values:
                        images = utils.normalize_input_values(images, "float32")
                    images, labels = convert_labels_to_time_pressed(images, labels)
                    images, labels = sequence_data(images, labels, shuffle_bool=False, incorporate_fps=incorporate_fps)
                    print(f"Epoch: {epoch}; Chunk {j+1} out of {filenames[i]['chunks']};"
                          f" {len(filenames) -  i} out of {len(filenames)} files to go!")
                    # test data is always last, meaning if next doesn't exists it's the test data
                    if not filenames[i]["test_data"]:
                        model.fit(images, labels, epochs=epoch+1, batch_size=config.BATCH_SIZE,
                                  class_weight=class_weights, initial_epoch=epoch, validation_split=0.1,
                                  shuffle=True)
                    else:
                        model.evaluate(images, labels, batch_size=config.BATCH_SIZE)
                    del images, labels
                    gc.collect()
            else:
                print(f"File {filenames[i]['filename']} existed at the beginning, not anymore!")
                continue
        if save_every_epoch:
            model.save(config.model_name + "_epoch_" + str(epoch))
    model.save(config.model_name + "_fully_trained")


def divide_dataset(filenames, allowed_ram_mb, test_data_size=0, normalize_input_values=True, incorporate_fps=True, known_normalize_growth=0):
    file_size_limit = allowed_ram_mb // config.sequence_len
    res_filenames = []
    for i in range(len(filenames) - 1, -1, -1):
        full_filename = current_data_dir + filenames[i]
        divider = {"filename": full_filename}
        labels = utils.load_file_only_labels(full_filename)
        # only load images if necessary
        file_size_mb = calc_filesize(full_filename, normalize_input_values, known_normalize_growth)
        if test_data_size <= 0:
            divider["test_data"] = False
        elif test_data_size < len(labels):
            full_seq_len = config.sequence_len * utils.get_fps_ratio() if incorporate_fps else config.sequence_len
            if test_data_size > full_seq_len:
                new_divider = div_test_data_helper(full_filename, test_data_size,
                                                   file_size_limit, normalize_input_values, is_test=True)
                res_filenames.insert(0, new_divider)
            if len(labels) - test_data_size > full_seq_len:
                new_divider = div_test_data_helper(full_filename, test_data_size,
                                                   file_size_limit, normalize_input_values, is_test=False)
                res_filenames.insert(0, new_divider)
            test_data_size = 0
            continue
        else:
            test_data_size -= len(labels)
            divider["test_data"] = True

        calc_chunks_and_indices(file_size_mb, file_size_limit, len(labels), divider)
        res_filenames.insert(0, divider)
    return res_filenames


def div_test_data_helper(filename, test_data_size, file_size_limit, normalize_input_values, is_test=True):
    m_byte = (1024 ** 2)
    new_divider = {"filename": filename, "test_data": is_test}
    images, labels = utils.load_file(filename)
    original_len = len(labels)
    if is_test:
        images, labels = images[-test_data_size:], labels[-test_data_size:]
    else:
        images, labels = images[:-test_data_size], labels[:-test_data_size]
    if normalize_input_values:
        images = utils.normalize_input_values(images, "float32")
    file_size_mb = images.nbytes // m_byte + labels.nbytes // m_byte
    offset = original_len - len(labels) if is_test else 0
    calc_chunks_and_indices(file_size_mb, file_size_limit, len(labels), new_divider, offset)
    return new_divider


def calc_chunks_and_indices(file_size_mb, file_size_limit, data_len, divider, offset=0):
    divider["chunks"] = file_size_mb // file_size_limit + 1
    step = data_len / divider["chunks"]
    divider["indices"] = [int(round(chunk * step)) + offset for chunk in range(divider["chunks"] + 1)]


def calc_filesize(full_filename, normalize, normalize_factor=0):
    m_byte = 1024 ** 2
    file_size_mb = os.stat(full_filename).st_size // m_byte
    if normalize:
        if normalize_factor == 0:
            images = utils.load_file_only_images(full_filename)
            images = utils.normalize_input_values(images, "float32")
            labels = utils.load_file_only_labels(full_filename)
            file_size_mb = images.nbytes // m_byte + labels.nbytes // m_byte
        else:
            file_size_mb = int(file_size_mb * normalize_factor)
    return file_size_mb


def sequence_data(data_x, data_y, shuffle_bool=True, incorporate_fps=True):
    if len(data_x) != len(data_y):
        ValueError(f"Data_x and Data_y length differ: Data_x:{len(data_x)}, Data_y:{len(data_y)}")
    images = []
    fps_ratio = utils.get_fps_ratio()
    if fps_ratio == 0 and incorporate_fps:
        raise ValueError('Fps ratio is 0, cannot divide by 0')
    full_seq_len = config.sequence_len * fps_ratio if incorporate_fps else config.sequence_len
    step = fps_ratio if incorporate_fps else 1

    if len(data_y) < full_seq_len:
        raise ValueError(f"Not enough data, minimum length should be {full_seq_len}, but is {len(data_y)}")

    for i in range(len(data_x) - full_seq_len + step):
        # i + full_seq_len is last NOT included index, therefore + step in for loop above
        images += [data_x[i:i+full_seq_len:step]]
    images = np.stack(images, axis=0)
    labels = data_y[full_seq_len-step:]
    # use keras fit shuffle, this creates a copy -> both arrays in ram for short time
    # also don't use if you use validation_split in fit (seen data as validation over multiple epochs)
    if shuffle_bool:
        images, labels = shuffle(images, labels)    # shuffle both the same way
    return images, labels


# not needed anymore
def subdivide_data(load_from, new_dir_name, chunks, keep_directory, test_data_size=None):
    # keep directory assumes the files are correct, will perform training as if they just got created
    if keep_directory and os.path.isdir(new_dir_name):
        print("Directory exists and was not changed, as specified")
        return
    data = np.load(load_from, allow_pickle=True)
    # data = data[:len(data) // 10]    # for testing
    name_for_file_path = new_dir_name + "/" + config.temp_data_chunk_name
    print("Data length:", len(data))
    print("Chunk length:", len(data) // chunks)

    if os.path.isdir(new_dir_name):
        remove_subdivided_data(new_dir_name)
    try:
        os.makedirs(new_dir_name)
    except OSError:
        print(f"Creation of the directory {new_dir_name} failed, probably because it already exists")
        return
    step = len(data) // chunks
    for i in range(chunks):
        # guarantee, that test data remains constant if specified
        if not test_data_size:
            np.save(name_for_file_path + str(i), data[step*i:step*(i+1)])
        else:
            # if test_data_size is bigger than step size, there will be less files than requested chunks
            # if test_data_size is smaller than step size, there will be one more file than requested chunks
            # when iterating over data later just check if next file exists, if no it means that it is the test data
            if (chunks - i - 1) * step >= test_data_size and i < chunks - 1:
                np.save(name_for_file_path + str(i), data[step*i:step*(i+1)])
            else:
                np.save(name_for_file_path + str(i), data[step*i:-test_data_size])
                np.save(name_for_file_path + str(i + 1), data[-test_data_size:])
                break
    del data


def remove_subdivided_data(dir_to_remove_name):
    if os.path.isdir(dir_to_remove_name):
        shutil.rmtree(dir_to_remove_name)
        print(f"Directory {dir_to_remove_name} successfully removed")
    else:
        print(f"Directory {dir_to_remove_name} not found")


def get_inverse_proportions(data):
    print(len(data))
    x = np.sum(data, axis=0)     # sum each label for each timestep separately
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    x = np.ones(x.shape) / x
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    print(x)
    print(x.shape)
    return x


def get_class_weights(test_data_size=0):
    labels = utils.load_labels_only(current_data_dir)
    # remove last x rows
    labels = np.concatenate(labels, axis=0)
    uesless, labels = convert_labels_to_time_pressed(range(100), labels)
    if test_data_size:
        labels = labels[:-test_data_size, :]
    labels = np.argmax(labels, axis=-1)
    classes = np.asarray(range(config.output_classes))
    inverse_proportions = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
    inverse_proportions = dict(enumerate(inverse_proportions))
    print("Proportions:", inverse_proportions)
    del labels
    return inverse_proportions


def test_sequence_data_no_mismatch():
    x = np.random.rand(1000, config.height, config.width, config.color_channels)
    y = np.random.rand(1000, config.output_classes)
    print("Images shape:", x.shape, "Labels shape:", y.shape)
    xx, yy = sequence_data(x, y, shuffle_bool=False, incorporate_fps=True)
    print("Images match?:", xx[-100][-1][0][0] == x[-100][0][0], "Labels match?:", yy[-100] == y[-100])
    del xx, yy
    xx, yy = sequence_data(x, y, shuffle_bool=False, incorporate_fps=False)
    print("Images match?:", xx[-100][-1][0][0] == x[-100][0][0], "Labels match?:", yy[-100] == y[-100])


def test_divide_dataset():
    filenames = utils.get_sorted_filenames(current_data_dir)
    if config.random_file_order_train:
        random.shuffle(filenames)
    filenames = divide_dataset(filenames, config.allowed_ram_mb, 10000, normalize_input_values=True,
                               incorporate_fps=False, known_normalize_growth=config.known_normalize_growth)
    for entry in filenames:
        print(entry)


def test_divide_cnn_only():
    filenames = utils.get_sorted_filenames(current_data_dir)
    if config.random_file_order_train:
        random.shuffle(filenames)
    my_dict = divide_dataset_cnn_only(filenames, 30000, True, normalize_factor=config.known_normalize_growth,
                                      allowed_ram=config.allowed_ram_mb)
    for i in my_dict:
        print(i)
    for i in my_dict:
        filesize = 0
        for j in i["filenames"]:
            filesize += os.stat(j).st_size
        print((filesize * 4) // (1024 ** 2))


def convert_labels_to_time_pressed(images, labels):
    fps_ratio = utils.get_fps_ratio()
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
    images = images[:-fps_ratio]
    return images, new_labels


if __name__ == "__main__":
    # train_model(True, freeze=True, load_saved_cnn=False)
    train_cnn_only(False, swap_output_layer=False, freeze_part=True)
