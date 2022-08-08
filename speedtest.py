import h5py
from tensorflow.keras.applications import EfficientNetB4, InceptionV3
from tensorflow.keras.models import load_model
import numpy as np
import time
import config
import lenet
import utils


def test_efficient_vs_inception():
    input_eff = 380
    input_incep = 299
    inception_v3 = InceptionV3(
        include_top=False,
        weights="imagenet",
        classes=config.output_classes,
        classifier_activation="softmax",
    )
    efficient_net_b4 = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        classes=config.output_classes,
        classifier_activation="softmax",
    )

    print("done building")

    for i in range(5):
        test_img_incep = np.random.uniform(-1, 1, size=(1, input_incep, input_incep, config.color_channels))
        t = time.time()
        inception_v3.predict(test_img_incep)
        print("inception prediction took:", time.time() - t)

    for i in range(5):
        test_img_eff = np.random.randint(255, size=(1, input_eff, input_eff, config.color_channels))
        t = time.time()
        efficient_net_b4.predict(test_img_eff)
        print("EFficientnet prediction took:", time.time() - t)


def predict_vs_other():
    input_eff = 380
    efficient_net_b4 = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        classes=config.output_classes,
        classifier_activation="softmax",
    )
    for i in range(20):
        test_img_eff = np.random.randint(255, size=(1, input_eff, input_eff, config.color_channels))
        t = time.time()
        print(efficient_net_b4.predict(test_img_eff).shape)
        print("EFficientnet prediction took:", time.time() - t)

    for i in range(20):
        test_img_eff = np.random.randint(255, size=(1, input_eff, input_eff, config.color_channels))
        t = time.time()
        print(efficient_net_b4(test_img_eff, training=False).numpy().shape)
        print("EFficientnet other method prediction took:", time.time() - t)


def check_preprocessing_slowdown():
    model1 = lenet.inception_with_preprocess_layer()
    model2 = lenet.create_cnn_only()    # no preprocessing
    amount = 30
    data = np.random.uniform(size=(amount, config.sequence_len, config.height, config.width, config.color_channels)) * 255
    for i in range(amount):
        t = time.time()
        model1.predict(data[i])
        print("With preprocessing prediction took:", time.time() - t)
    for i in range(amount):
        t = time.time()
        model2.predict(data[i])
        print("No preprocessing prediction took:", time.time() - t)


def lstm_only_pred_speed():
    model2 = lenet.lstm_only()
    amount = 30
    features = 2048
    data = np.random.uniform(size=(amount, 1, config.sequence_len, features))
    for i in range(amount):
        t = time.time()
        model2.predict(data[i])
        print("LSTM prediction took:", time.time() - t)


def test_combined_speed():
    model1 = lenet.inception_with_preprocess_layer()
    inputs = model1.input
    prep_inputs = lenet.preprocess_input(inputs)
    model1 = lenet.Model(inputs=model1.layers[-2].input, outputs=model1.layers[-2].output)(prep_inputs)
    model1 = lenet.Model(inputs=inputs, outputs=model1)
    model1.summary()
    model2 = lenet.lstm_only()
    amount = 30
    data = np.random.uniform(size=(amount, config.sequence_len, config.height, config.width, config.color_channels)) * 255
    for i in range(amount):
        t = time.time()
        # add single dim of 1 essentially
        feature_vector = model1.predict(data[i]).reshape(-1, config.sequence_len, 2048)
        final_prediction = model2.predict(feature_vector)
        print("With preprocessing prediction took:", time.time() - t)


def test_output_shape():
    output_arr = np.random.uniform(-1, 1, size=(1, config.output_classes))
    print(output_arr)
    print(output_arr.shape)
    print("-----------------")
    # output_arr = np.random.uniform(-1, 1, size=(1, 10, 6))   # 10 is sequence length, 6 is number of classes
    t = time.time()
    print("concat took:", time.time() - t)
    x = output_arr[0]
    print(x)
    x = np.argmax(x, axis=-1)
    print(x.shape)
    print(x)


def test_list_creating(reps):
    t = time.time()
    # fastest by far
    for i in range(reps):
        new_list = [j for j in range(100)]
    print("First method took:", time.time() - t)
    t = time.time()
    for i in range(reps):
        new_list = []
        for j in range(100):
            new_list.append(j)
    print("Second method took:", time.time() - t)
    t = time.time()
    for i in range(reps):
        new_list = []
        for j in range(100):
            new_list += [j]
    print("Third method took:", time.time() - t)


def test_if_still_np():
    x = np.random.rand(100, 3, 4, 2)
    sequence = x[0:50:5]
    k = [0]*100
    k = k[0:50]
    print(len(k))
    print(sequence.shape)
    print(sequence)
    sequence = sequence[:-1]
    print(sequence.shape)


def rewrite_dataset():
    for i in range(34):
        filename_load = config.turns_data_dir_name + config.data_name + f"_{i}.h5"
        filename_write = config.feature_extracted_data_name + config.data_name + f"_{i}.h5"
        data, labels = utils.load_file(filename_load)
        data = np.asarray(data)
        labels = np.asarray(labels)
        print(data.shape, labels.shape)
        with h5py.File(filename_write, 'w') as hf:
            hf.create_dataset("images", data=data)
            hf.create_dataset("labels", data=labels)
    """print("Now check load_speed")
    utils.load_file(config.feature_extracted_data_name + config.data_name + "_0sussy.h5")"""


def check_read_speed():
    for i in range(34):
        filename_load = config.feature_extracted_data_name + config.data_name + f"_{i}.h5"
        data, labels = utils.load_file(filename_load)
        data = np.asarray(data)
        labels = np.asarray(labels)
        print(data.shape, labels.shape)
        del data, labels


def write_feature_vector_data(model1=None):
    if model1 is None:
        model1 = lenet.inception_with_preprocess_layer()
    model1 = lenet.inception_expose_feature_layer(model1)
    model1.summary()
    data_dirs_to_convert = [config.new_data_dir_name, config.turns_data_dir_name, config.stuck_data_dir_name]
    filenames_to_convert = utils.get_sorted_filenames(data_dirs_to_convert)
    write_idx = 0
    for filename_read in filenames_to_convert:
        filename_write = config.feature_extracted_data_name + config.data_name + f"_{write_idx}.h5"
        write_idx += 1
        data, labels = utils.load_file(filename_read)
        data = np.asarray(data)
        labels = np.asarray(labels)
        data = model1.predict(data)
        print(data.shape, labels.shape)
        with h5py.File(filename_write, 'w') as hf:
            hf.create_dataset("images", data=data)
            hf.create_dataset("labels", data=labels)
        del data, labels


def test_if_batch_norm_working_after_freeze():
    model = load_model(config.cnn_only_name)
    model = lenet.unfreeze_inception(model, True)
    model.summary()
    data, labels = utils.load_file(config.new_data_dir_name + config.data_name + f"_0.h5")
    data, labels = utils.convert_labels_to_time_pressed(labels, images=data)
    predicted = model.predict(data)
    predicted = np.argmax(predicted, axis=-1)
    labels = np.argmax(labels, axis=-1)
    error = np.mean(labels != predicted)
    print(error)
    print(predicted.shape, labels.shape)
    print(labels[:100])


if __name__ == "__main__":
    model = load_model(config.cnn_only_name)
    # write_feature_vector_data(model)

