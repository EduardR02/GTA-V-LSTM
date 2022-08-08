import numpy as np
import tensorflow as tf
from lenet import lstm_only
from tensorflow import compat
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from lenet import inception_with_preprocess_layer, replace_cnn_dense_layer, freeze_part_of_inception, unfreeze_inception
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import time
import config
import utils
import random
import gc
import data_augmentation

current_data_dirs = [config.new_data_dir_name, config.turns_data_dir_name, config.stuck_data_dir_name]  # has to be list


def setup_tf():
    tf.keras.backend.clear_session()
    config_var = compat.v1.ConfigProto()
    config_var.gpu_options.per_process_gpu_memory_fraction = 1.0
    config_var.gpu_options.allow_growth = True
    compat.v1.InteractiveSession(config=config_var)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def train_model(load_saved):
    setup_tf()
    if load_saved:
        model = load_model(config.model_name)
    else:
        model = lstm_only(many_to_many=False)
    model.summary()
    random.seed(time.time())
    custom_training_loop(model, 40000, save_every_epoch=False, incorporate_fps=True,
                         shuffle=True, hold_data_in_mem=True, saved_file_order=False)


def train_cnn_only(load_saved, swap_output_layer=False, freeze_part=True):
    setup_tf()
    if load_saved:
        model = load_model(config.cnn_only_name)
        if swap_output_layer:
            model = replace_cnn_dense_layer(model)
    else:
        model = inception_with_preprocess_layer()
    if freeze_part:
        model = freeze_part_of_inception(model, "mixed9")
        # model = freeze_part_of_inception(model, "mixed10")  # full freeze
    else:
        model = unfreeze_inception(model)
    model.summary()
    cnn_only_training(model, 40000, shuffle=True)


def custom_training_loop(model, test_data_size, save_every_epoch, incorporate_fps=True, shuffle=True,
                         hold_data_in_mem=False, saved_file_order=False):
    """log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)"""
    class_weights = utils.get_class_weights(current_data_dirs, test_data_size=0)
    if not saved_file_order:
        filenames = utils.get_sorted_filenames(current_data_dirs)
        if config.random_file_order_train:
            random.shuffle(filenames)
        filename_dict_list = utils.divide_dataset_lstm_compatible(filenames, test_data_size, config.allowed_ram_mb,
                                                                  incorporate_fps=incorporate_fps)
    else:
        filename_dict_list = utils.load_training_file_list("data/train_data_dict_list_lstm.txt")
    dataset_list = []
    if hold_data_in_mem:
        dataset_list = gen_data_from_dict_list(filename_dict_list, incorporate_fps, shuffle)
    for k in filename_dict_list:
        print(k)
    for epoch in range(config.epochs):
        for i in range(len(filename_dict_list)):
            K.clear_session()
            if hold_data_in_mem:
                sequenced_data = dataset_list[i]
            else:
                sequenced_data = get_sequenced_data(filename_dict_list[i], incorporate_fps, shuffle=shuffle)
            print(f"Epoch: {epoch}; Files: {len(filename_dict_list[i]['filenames'])};"
                  f" {len(filename_dict_list) - i} out of {len(filename_dict_list)} file groups to go!")
            # test data is always last, meaning if next doesn't exists it's the test data
            if not filename_dict_list[i]["is_test"]:
                # validation does not make sense if you shuffle in the generator
                model.fit(sequenced_data, epochs=epoch + 1,
                          # class_weight=class_weights,
                          initial_epoch=epoch, shuffle=shuffle
                          # validation_split=0.1
                          )
            else:
                model.evaluate(sequenced_data, batch_size=config.BATCH_SIZE)
            if not hold_data_in_mem:
                del sequenced_data
            gc.collect()
        if save_every_epoch:
            model.save(config.model_name + "_epoch_" + str(epoch))
        elif epoch % 5 == 0 and epoch != 0:
            model.save(config.model_name + "_epoch_" + str(epoch))
    model.save(config.model_name + "_fully_trained")


def cnn_only_training(model, test_data_size, shuffle=True):
    # class_weights = utils.get_class_weights(current_data_dirs, test_data_size=test_data_size, convert_time_pressed=False)
    filenames = utils.get_sorted_filenames(current_data_dirs)
    if shuffle:
        random.shuffle(filenames)
    filename_dict_list = utils.divide_dataset_lstm_compatible(filenames, test_data_size,
                                                              allowed_ram=config.allowed_ram_mb)
    filename_dict_list = utils.load_training_file_list("data/train_data_dict_list_cnn.txt")
    for k in filename_dict_list:
        print(k)
    for epoch in range(config.epochs):
        for filename_dict in filename_dict_list:
            K.clear_session()
            images, labels = utils.concat_data_from_dict(filename_dict)
            labels = utils.convert_labels_to_binary(labels)
            labels, images = utils.convert_bin_labels_to_mse(labels, images=images)
            class_weights = utils.get_class_weights("doesnt matter", labels=labels, convert_time_pressed=False)
            if epoch == 0:
                print("class weights for filenames:", class_weights)
            # images, labels = utils.convert_labels_to_time_pressed(labels, images=images)
            if not filename_dict["is_test"]:
                dataset = data_augmentation.get_augmented_dataset(images, labels)
                model.fit(dataset,
                          epochs=epoch + 1, initial_epoch=epoch,
                          class_weight=class_weights,
                          )
                del dataset
            else:
                with tf.device("CPU:0"):
                    images, labels = tf.convert_to_tensor(images), tf.convert_to_tensor(labels)
                # mimic training loss function
                sample_weights = utils.generate_sample_weights_from_class_weight_dict(labels, class_weights)
                model.evaluate(images, labels, batch_size=config.CNN_ONLY_BATCH_SIZE, sample_weight=sample_weights)
                print("augmented eval:")
                dataset = data_augmentation.get_augmented_dataset(images, labels)
                model.evaluate(dataset)     # sample weight not supported when using dataset ( cringe )
                del dataset
            del images, labels
            gc.collect()
        # if epoch % 5 == 0 and epoch != 0:
        model.save(config.cnn_only_name + "_epoch_n_" + str(epoch))
    model.save(config.cnn_only_name)


def gen_data_from_dict_list(dict_list, incorporate_fps, shuffle=True):
    res_list = []
    for filename_dict in dict_list:
        res_list += [get_sequenced_data(filename_dict, incorporate_fps, shuffle)]
    return res_list


def get_sequenced_data(filename_dict, incorporate_fps, shuffle=True):
    # don't concat to np arr cuz first need to seq
    images, labels = utils.concat_data_from_dict(filename_dict, concat=False)
    combined_seq_data = None
    for i in range(len(labels)):
        images_curr, labels_curr = utils.convert_labels_to_time_pressed(labels[i], images=images[i])
        curr_seq_data = generate_timeseries(images_curr, labels_curr, shuffle=shuffle, incorporate_fps=incorporate_fps)
        if not combined_seq_data:
            combined_seq_data = curr_seq_data
        else:
            combined_seq_data = combined_seq_data.concatenate(curr_seq_data)
    return combined_seq_data


def generate_timeseries(images, labels, shuffle=False, incorporate_fps=True):
    sampling_rate = utils.get_fps_ratio() if incorporate_fps else 1
    # labels have to correspond to predicted sequence, not samplerate-1 because we want the next timestep as label
    labels = labels[sampling_rate * config.sequence_len:]
    sequenced_data = timeseries_dataset_from_array(images, labels, sequence_length=config.sequence_len,
                                                   sampling_rate=sampling_rate,
                                                   sequence_stride=config.sequence_stride,
                                                   batch_size=config.BATCH_SIZE,
                                                   shuffle=shuffle)
    return sequenced_data


def test_generate_time_series():
    images, labels = utils.load_file(config.stuck_data_dir_name + config.data_name + "_0.h5")
    print(len(labels))
    data = generate_timeseries(images, labels, shuffle=False, incorporate_fps=True)
    shift_val = utils.get_fps_ratio() * config.sequence_len
    print(len(labels) - shift_val)
    fps_ratio = utils.get_fps_ratio()
    for i, batch in enumerate(data):
        ninputs, nlabels = batch
        print(len(nlabels), len(ninputs))
        for j in range(len(nlabels)):
            temp_l = labels[j + shift_val + i * config.BATCH_SIZE]
            temp_i = images[i * config.BATCH_SIZE + j: i * config.BATCH_SIZE + shift_val + j: fps_ratio]
            print(i * config.BATCH_SIZE + j)
            assert np.array_equal(ninputs[j], temp_i)
            assert np.array_equal(nlabels[j], temp_l)


if __name__ == "__main__":
    # train_model(load_saved=False)
    train_cnn_only(load_saved=False, swap_output_layer=False, freeze_part=True)
