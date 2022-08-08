import numpy as np
import time
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, Input, RepeatVector, TimeDistributed, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import config
import utils


# height first, then width is keras order
def create_neural_net(load_pretrained_cnn=False, model_name=""):
    np.random.seed(1000)
    inputs = Input(shape=(config.sequence_len, config.height, config.width, config.color_channels))
    prep_inputs = preprocess_input(inputs)
    if load_pretrained_cnn:
        if model_name == "": raise ValueError("model_name cannot be empty")
        cnn = load_model(model_name)
        # strips cnn of everything but inception with 2048 avgpool output, layers[...] is where that layer is
        cnn = Model(inputs=cnn.layers[1].input, outputs=cnn.layers[1].output)
    else:
        cnn = InceptionV3(weights="imagenet", include_top=False,
                          pooling="avg", input_shape=(config.height, config.width, config.color_channels))
        cnn = Model(inputs=cnn.input, outputs=cnn.output)
    # cnn.trainable = False     # testing
    x = TimeDistributed(cnn)(prep_inputs)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, input_shape=(config.sequence_len,
                               (config.height, config.width, config.color_channels)), return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # IDEA: use sigmoid output with binary cross entropy, meaning each output can be pressed independently
    # possible, just convert wa, wd to w and d and w and a, otherwise the same
    outputs = Dense(config.output_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=config.lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def many_to_many_lstm():
    timesteps_to_predict = utils.get_fps_ratio()
    features = 2048     # inception feature vector
    print(timesteps_to_predict)
    inputs = Input(shape=(config.sequence_len, config.height, config.width, config.color_channels))
    cnn = InceptionV3(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(config.height, config.width, config.color_channels))
    cnn = Model(inputs=cnn.input, outputs=cnn.output)
    prep_inputs = preprocess_input(inputs)
    x = TimeDistributed(cnn)(prep_inputs)
    x = TimeDistributed(Flatten())(x)
    # encoder decoder lstm, with output being defined amount of timesteps
    x = LSTM(256, input_shape=(config.sequence_len, features))(x)
    x = RepeatVector(timesteps_to_predict)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = TimeDistributed(Dropout(0.2))(x)
    model_outputs = TimeDistributed(Dense(config.output_classes, activation="softmax"))(x)
    model = Model(inputs, model_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


# Use other one please, the one that has preprocessing
def create_cnn_only():
    np.random.seed(1000)
    inputs = Input(shape=(config.height, config.width, config.color_channels))
    cnn = InceptionV3(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(config.height, config.width, config.color_channels))
    cnn = cnn(inputs)
    # cnn = Dropout(0.3)(cnn)
    cnn = Dense(config.output_classes, activation="softmax")(cnn)
    cnn = Model(inputs=inputs, outputs=cnn)
    optimizer = Adam(learning_rate=config.cnn_lr)
    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn


def replace_cnn_dense_layer(model):
    """
    Probably won't include the preprocessing layer if one is present
    """
    np.random.seed(1000)
    inputs = model.input
    # prep_inputs = preprocess_input(inputs)
    # model = Model(inputs=model.layers[-2].input, outputs=model.layers[-2].output)(prep_inputs)
    model = Dense(config.output_classes, activation="softmax")(model.layers[-2].output)
    model = Model(inputs=inputs, outputs=model)
    optimizer = Adam(learning_rate=config.cnn_lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def test_model_speed():
    name = "car_2_inception_and_lstm_normalized_more_dropout"
    # my_model = load_model(name)
    my_model = create_neural_net()
    my_model.summary()
    for i in range(6):
        t = time.time()
        my_model.predict(np.random.randint(255, size=(1, config.sequence_len, config.height,
                                                      config.width, config.color_channels)))
        print(time.time() - t)


def save_untrained_model():
    name = "model_for_eff_net_test_fps"
    model = create_neural_net()
    model.summary()
    model.save(name)


def freeze_part_of_inception(model, layer_name="mixed9"):
    # mixed8, mixed9
    # this function is for a very specific model architecture, bunch of "magic" numbers
    idx = 0
    inception_layer = 3     # magic number, just check manually
    while True:     # is supposed to throw if specified name does not exist
        if model.layers[inception_layer].layers[idx].name == layer_name: break
        model.layers[inception_layer].layers[idx].trainable = False
        idx += 1
    optimizer = Adam(learning_rate=config.cnn_lr)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    # model.layers[inception_layer].summary()
    return model


def unfreeze_inception(model, full_unfreeze=False):
    # do not unfreeze batch norms that have been frozen, as by unfreezing they will have completely different values,
    # because when frozen they use the saved mean, which will be different from the current running mean
    for layer in model.layers[-3].layers:
        # yes you could just restructure the if to remove the else, but this conveys the purpose better
        if not full_unfreeze and isinstance(layer, BatchNormalization) and not layer.trainable:
            layer.trainable = False
        else:
            layer.trainable = True
    optimizer = Adam(learning_rate=config.cnn_lr)
    # model.layers[-2].summary()
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    return model


def lstm_only(many_to_many=False):
    """
    OBSERVATIONS:
        -1024 lstm first layer won't work because of mem limitation (inception is in mem too),
            also seems to overfit quite hard and slow in general
        -below 128, maybe even 256 first layer seems to under fit.
        -Batch Norm between lstm layers helps a lot, single bnorm after first lstm seems to perform best, even if
            3 layers of lstm.
        -Two lstm layers seem to perform better than one,
            might need to retest to check if one without batch norm is better
        -Three layers don't seem to perform as well, not many values tried tho
        -First layer is good with 512 or 256, second layer minimize overfitting, 64 and 128 respectfully,
            values between given values in both layers could work better
    """
    # specific to inception feature output
    np.random.seed(1000)
    features = 2048
    timesteps_to_predict = utils.get_fps_ratio()
    inputs = Input(shape=(config.sequence_len, features))
    x = LSTM(512, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    if many_to_many:
        x = RepeatVector(timesteps_to_predict)(x)
    x = LSTM(256, return_sequences=many_to_many)(x)
    if many_to_many:
        model_outputs = TimeDistributed(Dense(config.output_classes, activation="softmax"))(x)
    else:
        model_outputs = Dense(config.output_classes, activation="softmax")(x)
    model = Model(inputs, model_outputs)
    optimizer = Adam(learning_rate=config.lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def inception_with_preprocess_layer():
    np.random.seed(1000)
    inputs = Input(shape=(config.height, config.width, config.color_channels))
    prep_inputs = preprocess_input(inputs)
    cnn = InceptionV3(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(config.height, config.width, config.color_channels))
    cnn = cnn(prep_inputs)
    cnn = Dropout(0.3)(cnn)
    # when using mae or mse loss use the relu that maxes out at 1.0
    cnn = Dense(len(config.outputs_base), activation=relu_limited)(cnn)
    cnn = Model(inputs=inputs, outputs=cnn)
    optimizer = Adam(learning_rate=config.cnn_lr)
    cnn.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    return cnn


def relu_limited(x):
    return K.relu(x, max_value=1.0)


def test_lstm_only():
    model = lstm_only()
    model.summary()
    features = 2048
    dataset_size = 50
    timesteps_per_prediction = utils.get_fps_ratio()
    labels = to_categorical(
        np.random.randint(0, config.output_classes, size=(dataset_size, timesteps_per_prediction, 1)))
    print(labels.shape)
    print(labels[0])
    inputs = np.random.uniform(size=(dataset_size, config.sequence_len, features))
    model.fit(inputs, labels, epochs=2, batch_size=128, validation_split=0.1, shuffle=False)


def inception_expose_feature_layer(model):
    inputs = model.input
    prep_inputs = preprocess_input(inputs)
    # insert actual model here, this is just for testing
    model = Model(inputs=model.layers[-2].input, outputs=model.layers[-2].output)(prep_inputs)
    model = Model(inputs=inputs, outputs=model)
    return model


if __name__ == "__main__":
    """model = create_cnn_only()
    model = freeze_part_of_inception(model, "mixed9")
    model.summary()"""
    # test_lstm_only()
    model = inception_with_preprocess_layer()
    model.summary()
    amount = 30
    x = np.random.uniform(size=(amount, config.height, config.width, config.color_channels)) * 255
    print(model.predict(x))



