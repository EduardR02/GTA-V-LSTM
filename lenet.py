from tensorflow.keras.layers import Dense, LSTM, Input
import numpy as np
import time
from tensorflow.keras.applications import EfficientNetB3, InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, Flatten, Dropout
from tensorflow.keras.models import load_model
import config


# height first, then width is keras order
def create_neural_net(load_pretrained_cnn=False, model_name=""):
    np.random.seed(1000)
    inputs = Input(shape=(config.sequence_len, config.height, config.width, config.color_channels))
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
    x = TimeDistributed(cnn)(inputs)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, input_shape=(config.sequence_len,
                               (config.height, config.width, config.color_channels)), return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    outputs = Dense(config.output_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=config.lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_cnn_only():
    np.random.seed(1000)
    inputs = Input(shape=(config.height, config.width, config.color_channels))
    cnn = InceptionV3(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(config.height, config.width, config.color_channels))
    cnn = cnn(inputs)
    cnn = Dropout(0.3)(cnn)
    cnn = Dense(config.output_classes, activation="softmax")(cnn)
    cnn = Model(inputs=inputs, outputs=cnn)
    optimizer = Adam(learning_rate=config.lr)
    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn


def replace_cnn_dense_layer(model):
    np.random.seed(1000)
    inputs = model.input
    # cut off output layer and replace with new one
    model = Dense(config.output_classes, activation="softmax")(model.layers[-2].output)
    model = Model(inputs=inputs, outputs=model)
    optimizer = Adam(learning_rate=config.lr)
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
    while True:     # is supposed to throw if specified name does not exist
        if model.layers[1].layers[idx].name == layer_name: break
        model.layers[1].layers[idx].trainable = False
        idx += 1
    optimizer = Adam(learning_rate=config.lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = create_cnn_only()
    model = freeze_part_of_inception(model, "mixed9")
    model.summary()



