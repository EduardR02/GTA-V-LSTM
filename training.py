import numpy as np
import tensorflow as tf
from lenet import create_neural_net, create_cnn_only
from tensorflow import compat
import time
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import gc
import os
import shutil
import datetime
import config


def setup_tf():
    tf.keras.backend.clear_session()
    config = compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    compat.v1.Session(config=config)
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
    custom_training_loop(model, 50, 5000, True, False, keep_dir=True)


def train_cnn_only(load_saved):
    setup_tf()
    if load_saved:
        model = load_model(config.cnn_only_name)
    else:
        model = create_cnn_only()
    model.summary()
    cnn_only_training(model)


def cnn_only_training(model):
    class_weights = get_class_weights(test_data_size=10000)
    t = time.time()
    train_data = np.load(config.load_data_name, allow_pickle=True)
    print("Data loaded in:", time.time() - t, "seconds.")

    # train_data = train_data[:len(train_data) // 10]

    print("Training data amount:", len(train_data))

    train = train_data[:-10000]
    validation = train_data[-10000:-5000]
    test = train_data[-5000:]
    # delete lists when already processed to speed everything up when memory limited
    del train_data
    t = time.time()
    # height, width is the correct reshape order!!!!!!!!!

    label_data_train = np.array([i[1] for i in train]).reshape((-1, config.output_classes))
    image_data_train = np.array([i[0] for i in train]).reshape((-1, config.height, config.width, config.color_channels))
    print(label_data_train.shape, image_data_train.shape)
    del train

    label_data_validation = np.array([i[1] for i in validation]).reshape((-1, config.output_classes))
    image_data_validation = np.array([i[0] for i in validation]).reshape((-1, config.height,
                                                                          config.width, config.color_channels))
    del validation

    label_data_test = np.array([i[1] for i in test]).reshape((-1, config.output_classes))
    image_data_test = np.array([i[0] for i in test]).reshape((-1, config.height, config.width, config.color_channels))
    del test

    print("Data reshaped in:", time.time() - t, "seconds.")

    model.fit(image_data_train, label_data_train,
              epochs=config.eps, batch_size=config.BATCH_SIZE, class_weight=class_weights,
              validation_data=(image_data_validation, label_data_validation), shuffle=True)
    model.evaluate(image_data_test, label_data_test)
    model.save(config.cnn_only_name)


def custom_training_loop(model, chunks, test_data_size, save_every_epoch, halfway_save, keep_dir=False):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    class_weights = get_class_weights(test_data_size=test_data_size)
    # prepare chunks and save them
    subdivide_data(load_from=config.load_data_name, new_dir_name=config.temp_data_folder_name,
                   chunks=chunks, keep_directory=keep_dir, test_data_size=test_data_size)
    name_for_file_path = config.temp_data_folder_name + "/" + config.temp_data_chunk_name
    for i in range(config.eps):
        K.clear_session()
        # + 1 as data could be split into +1 chunk because of test data size
        for current_chunk in range(chunks + 1):
            K.clear_session()
            # check if file exists, if not this epoch is done
            if os.path.isfile(name_for_file_path + str(current_chunk) + ".npy"):
                data = np.load(name_for_file_path + str(current_chunk) + ".npy", allow_pickle=True)
                images, labels = sequence_data(data, shuffle_bool=False, incorporate_fps=True)
                del data
                print("Epoch: {}; Chunk {} out of {}".format(i, current_chunk, chunks))
                # test data is always last, meaning if next doesn't exists it's the test data
                if os.path.isfile(name_for_file_path + str(current_chunk + 1) + ".npy"):
                    # epochs is i+1 because we want to train only one iteration, initial epoch is "starting epoch"
                    # if initial_epoch == epochs then it doesn't train  (just skips)
                    model.fit(images, labels, epochs=i+1, batch_size=config.BATCH_SIZE,
                              class_weight=class_weights, initial_epoch=i, validation_split=0.1,
                              callbacks=[tensorboard_callback], shuffle=True)
                else:
                    model.evaluate(images, labels)
                del images, labels
            else:
                break
            # save at half if one epoch takes too long
            if current_chunk == chunks // 2 and halfway_save and save_every_epoch:
                model.save(config.model_name + "_epoch_" + str(i) + "_halfway")
            gc.collect()
        if save_every_epoch: model.save(config.model_name + "_epoch_" + str(i))
        gc.collect()
    model.save(config.model_name + "_fully_trained")
    # delete temporary created chunks
    remove_subdivided_data(config.temp_data_folder_name)


def sequence_data(data, shuffle_bool=True, incorporate_fps=True):
    res = []
    if len(data) < config.sequence_len:
        print("Too little data")
        return
    # split data for memory efficiency
    images = data[:, 0]
    labels = data[:, 1]
    del data
    # approximate test time fps into training data
    if not incorporate_fps:
        if len(images) < config.sequence_len:
            raise ValueError("Not enough data, minimum length should be {}, but is {}"
                             .format(config.sequence_len, len(images)))

        for i in range(len(images) - config.sequence_len + 1):
            res += list(images[i:i+config.sequence_len])    # select sequence length

        images = np.asarray(res).reshape((-1, config.sequence_len, config.height, config.width, config.color_channels))
        del res
        # need to match labels to last image in sequence
        labels = np.concatenate(labels[config.sequence_len - 1:], axis=0).reshape((-1, config.output_classes))
    else:
        # determines how many frames to skip
        fps_ratio = int(round(config.fps_at_recording_time / config.fps_at_test_time))
        if fps_ratio == 0: raise ValueError('Fps ratio is 0, cannot divide by 0')
        if len(images) < config.sequence_len * fps_ratio:
            raise ValueError("Not enough data, minimum length should be {}, but is {}"
                             .format(config.sequence_len*fps_ratio, len(images)))

        for i in range(len(images) - config.sequence_len * fps_ratio + 1):
            res += list(images[i: i + (fps_ratio*config.sequence_len): fps_ratio])

        images = np.asarray(res).reshape((-1, config.sequence_len, config.height, config.width, config.color_channels))
        del res
        labels = np.concatenate(labels[(fps_ratio*config.sequence_len) - 1:], axis=0).reshape((-1, config.output_classes))
    # use keras fit shuffle, this creates a copy -> both arrays in ram for short time
    # also don't use if you use validation_split in fit,
    # as it will kill the purpose (seen data as validation over multiple epochs)
    if shuffle_bool:
        images, labels = shuffle(images, labels)    # shuffle both the same way
    gc.collect()
    print("Image array shape:", images.shape)
    print("Label array shape", labels.shape)
    return images, labels


# creates a new directory with divided dataset into smaller chunks
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
    data = np.load(config.load_data_name, allow_pickle=True)
    labels = data[:, 1]
    del data
    gc.collect()
    labels = labels[:-test_data_size]
    labels = [y.index(max(y)) for y in labels]
    inverse_proportions = class_weight.compute_class_weight('balanced',
                                                            classes=np.unique(labels),
                                                            y=labels)
    inverse_proportions = dict(enumerate(inverse_proportions))
    print("Proportions:", inverse_proportions)
    del labels
    gc.collect()
    return inverse_proportions


if __name__ == "__main__":
    train_model(False, freeze=True, load_saved_cnn=True)
    # train_cnn_only(True)
