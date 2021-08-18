import numpy as np
import tensorflow as tf
from lenet import create_neural_net
from tensorflow import compat
import time
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import gc
import os
import shutil

width = 160
height = 120
lr = 5e-5
color_channels = 3
eps = 5
model_dir_name = "models/"
model_name = model_dir_name + "car_inception_and_lstm_fps_adjusted_v6.5.1.2_epoch_8"
load_data_name = "training_data_for_lstm_rgb_full.npy"
sequence_len = 20
output_classes = 6
BATCH_SIZE = 24     # depends a lot on hardware, but also can be much higher if part of the model is frozen
temp_data_chunk_name = "temp_dataset_chunk_"
temp_data_folder_name = "data_in_chunks_temp"
# around 80 for me depending on population density in game
fps_at_recording_time = 80      # check by using main with fps only set to true, while having the game running
# check by running model in main, i get about about 9 - 10,
# has to be lower than recording-time-fps (else you might get division by 0. If that is the case just don't use this)
fps_at_test_time = 9   # that might change with model architecture though (like using efficientnet over inception)


def train_model(load_saved, freeze=False):
    tf.keras.backend.clear_session()
    config = compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    compat.v1.Session(config=config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if load_saved:
        model = load_model(model_name)
    else:
        model = create_neural_net(height, width, lr, color_channels, sequence_len)
    # freeze convolutional model to fine tune lstm (the cnn is treated as one layer
    # make sure you freeze the correct one)
    # goes the other way around too if the model was saved frozen and you want to unfreeze
    model.layers[1].trainable = not freeze
    # you might want to lower the learning rate and increase training epochs when frozen
    optimizer = Adam(learning_rate=lr)
    # recompile to make the changes
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    custom_training_loop(model, 50, 5000, True, False, keep_dir=True)


# with the custom loop implemented this is pretty useless, as it is much worse
def standard_training_loop(model):
    # get data
    t = time.time()
    train_data = np.load(load_data_name, allow_pickle=True)
    print("Data loaded in:", time.time() - t, "seconds.")

    # train_data = train_data[:len(train_data) // 3]

    print("Training data amount:", len(train_data))
    train = train_data[:-5000]
    validation = train_data[-5000:-2500]
    test = train_data[-2500:]
    # delete lists when already processed to speed everything up when memory limited
    del train_data
    t = time.time()
    # height, width is the correct reshape order!!!!!!!!!

    image_data_train, label_data_train = sequence_data(train)
    del train
    image_data_validation, label_data_validation = sequence_data(validation)
    del validation
    image_data_test, label_data_test = sequence_data(test)
    del test

    print("Data reshaped in:", time.time() - t, "seconds.")

    model.fit(image_data_train, label_data_train,
              epochs=eps, batch_size=BATCH_SIZE,
              validation_data=(image_data_validation, label_data_validation), shuffle=True)
    model.evaluate(image_data_test, label_data_test)
    model.save(model_name)


# use when ram is limited
def custom_training_loop(model, chunks, test_data_size, save_every_epoch, halfway_save, keep_dir=False):
    global lr
    # get inverse proportions of the classes
    class_weights = get_class_weights()
    # prepare chunks and save them
    subdivide_data(load_from=load_data_name, new_dir_name=temp_data_folder_name,
                   chunks=chunks, keep_directory=keep_dir, test_data_size=test_data_size)
    name_for_file_path = temp_data_folder_name + "/" + temp_data_chunk_name
    for i in range(eps):
        K.clear_session()
        # not always useful, maybe disable it (because the optimizer should already handle this)
        if i != 0:
            lr = lr / 2
            K.set_value(model.optimizer.learning_rate, lr)
        # + 1 because if the test data is smaller than the chunks,
        # the last chunk will be split into a smaller chunk and the test data, meaning there is one more file
        for current_chunk in range(chunks + 1):
            K.clear_session()
            # check if file exists, if not this epoch is done
            if os.path.isfile(name_for_file_path + str(current_chunk) + ".npy"):
                data = np.load(name_for_file_path + str(current_chunk) + ".npy", allow_pickle=True)
                images, labels = sequence_data(data, shuffle_bool=True, incorporate_fps=True)
                del data
                print("Epoch: {}; Chunk {} out of {}".format(i, current_chunk, chunks))
                # check if data is test data or train data,
                # test is always last, meaning if next doesn't exists it's the test data
                if os.path.isfile(name_for_file_path + str(current_chunk + 1) + ".npy"):
                    # epochs is i+1 because we want to train only one iteration, initial epoch is "starting epoch"
                    # so if initial_epoch == epochs then it doesn't train  (just skips)
                    model.fit(images, labels, epochs=i+1, batch_size=BATCH_SIZE,
                              class_weight=class_weights, initial_epoch=i)
                else:
                    # validation
                    model.evaluate(images, labels)
                del images, labels
            else:
                gc.collect()
                break
            # save at half if one epoch takes too long
            if current_chunk == chunks // 2 and halfway_save and save_every_epoch:
                model.save(model_name + "_epoch_" + str(i) + "_halfway")
            gc.collect()
        if save_every_epoch: model.save(model_name + "_epoch_" + str(i))
        gc.collect()
    model.save(model_name + "_fully_trained")
    # delete temporary created chunks
    remove_subdivided_data(temp_data_folder_name)


def sequence_data(data, shuffle_bool=True, incorporate_fps=True):
    res = []
    if len(data) < sequence_len:
        print("Too little data")
        return
    # split data for memory efficiency
    images = data[:, 0]
    labels = data[:, 1]
    del data
    # incorporate_fps means to take into account training time fps, as collecting data has higher fps than when
    # the model is predicting. If inc_fps is True, the data will be approximated to test time fps
    if not incorporate_fps:
        # correct length to fit sequence len
        if len(images) < sequence_len:
            raise ValueError("Not enough data, minimum length should be {}, but is {}"
                             .format(sequence_len, len(images)))

        for i in range(len(images) - sequence_len + 1):
            res += list(images[i:i+sequence_len])    # select sequence length

        images = np.array(res).reshape((-1, sequence_len, height, width, color_channels))
        del res
        # need to match labels start and last image in sequence
        # for example if seq len is 20, then first sequence end will be index 19
        # concatenate because all labels are lists, not np arrays
        labels = np.concatenate(labels[sequence_len - 1:], axis=0).reshape((-1, output_classes))
    else:
        fps_ratio = int(round(fps_at_recording_time / fps_at_test_time))    # determines how many frames to skip
        if fps_ratio == 0: raise ValueError('Fps ratio is 0, cannot divide by 0')
        if len(images) < sequence_len * fps_ratio:
            raise ValueError("Not enough data, minimum length should be {}, but is {}"
                             .format(sequence_len*fps_ratio, len(images)))

        for i in range(len(images) - sequence_len * fps_ratio + 1):
            res += list(images[i: i + (fps_ratio*sequence_len): fps_ratio])

        images = np.array(res).reshape((-1, sequence_len, height, width, color_channels))
        del res
        labels = np.concatenate(labels[(fps_ratio*sequence_len) - 1:], axis=0).reshape((-1, output_classes))
    if shuffle_bool:
        images, labels = shuffle(images, labels)    # shuffle both the same way
    print("Image array shape:", images.shape)
    print("Label array shape", labels.shape)
    return images, labels


# creates a new directory with divided dataset into smaller chunks
def subdivide_data(load_from, new_dir_name, chunks, keep_directory, test_data_size=None):
    # keep directory assumes the files are correct, will perform training as if they just got created, so be careful
    if keep_directory and os.path.isdir(new_dir_name):
        print("Directory exists and was not changed, as specified")
        return
    # makes it so last file is only test data
    data = np.load(load_from, allow_pickle=True)
    # data = data[:len(data) // 10]    # for testing
    name_for_file_path = new_dir_name + "/" + temp_data_chunk_name
    print("Data length:", len(data))
    print("Chunk length:", len(data) // chunks)
    # remove directory if it exists.
    if os.path.isdir(new_dir_name):
        remove_subdivided_data(new_dir_name)
    try:
        os.makedirs(new_dir_name)
    except OSError:
        print("Creation of the directory %s failed, probably because it already exists" % new_dir_name)
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
        print("Directory %s successfully removed" % dir_to_remove_name)
    else:
        print("Directory %s not found" % dir_to_remove_name)


# class weights not supported on lstm rip, sample weights are though but I don't think I want to use them
def get_inverse_proportions(data):
    print(len(data))
    x = np.sum(data, axis=0)     # sum each label for each timestep separately
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    x = np.ones(x.shape) / x
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    print(x)
    print(x.shape)
    return x


def get_class_weights():
    data = np.load(load_data_name, allow_pickle=True)
    labels = data[:, 1]
    del data
    labels = [y.index(max(y)) for y in labels]
    inverse_proportions = class_weight.compute_class_weight('balanced',
                                                            classes=np.unique(labels),
                                                            y=labels)
    inverse_proportions = dict(enumerate(inverse_proportions))
    print("Proportions:", inverse_proportions)
    return inverse_proportions


if __name__ == "__main__":
    train_model(True, True)
