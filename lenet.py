from tensorflow.keras.layers import Dense, LSTM, Input
import numpy as np
import time
from tensorflow.keras.applications import EfficientNetB3, InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, Flatten, Dropout
from tensorflow.keras.models import load_model


# height first, then width is keras order
def create_neural_net(height, width, lr, color_channels, sequence_length, load_pretrained_cnn=False, model_name=""):
    np.random.seed(1000)
    inputs = Input(shape=(sequence_length, height, width, color_channels))
    if load_pretrained_cnn:
        if model_name == "": raise ValueError("model_name cannot be empty")
        cnn = load_model(model_name)
        # strips cnn of everything but inception with 2048 avgpool output, layers[...] is where that layer is
        cnn = Model(inputs=cnn.layers[1].input, outputs=cnn.layers[1].output)
    else:
        cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(height, width, color_channels))
        cnn = Model(inputs=cnn.input, outputs=cnn.output)
    # cnn.trainable = False     # testing
    x = TimeDistributed(cnn)(inputs)
    x = TimeDistributed(Flatten())(x)

    x = LSTM(512, input_shape=(sequence_length, (height, width, color_channels)), return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    outputs = Dense(6, activation="softmax")(x)
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_cnn_only(height, width, lr, color_channels):
    np.random.seed(1000)
    inputs = Input(shape=(height, width, color_channels))
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(height, width, color_channels))
    cnn = cnn(inputs)
    cnn = Dropout(0.5)(cnn)
    cnn = Dense(6, activation="softmax")(cnn)
    cnn = Model(inputs=inputs, outputs=cnn)
    optimizer = Adam(learning_rate=lr)
    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return cnn


def test_model_speed():
    name = "car_2_inception_and_lstm_normalized_more_dropout"
    my_model = load_model(name)
    my_model.summary()
    t = time.time()
    test_img_eff = np.random.randint(255, size=(1, 20, 120, 160, 3))
    print(my_model.predict(test_img_eff))
    print("Model took:", time.time() - t)
    t = time.time()
    test_img_eff = np.random.randint(255, size=(1, 20, 120, 160, 3))
    print(my_model.predict(test_img_eff))
    print("Model took:", time.time() - t)
    t = time.time()
    test_img_eff = np.random.randint(255, size=(1, 20, 120, 160, 3))
    x = my_model.predict(test_img_eff)
    print(x)
    print("Model took:", time.time() - t)
    print("x output shape is:", x.shape)


def save_untrained_model():
    name = "model_for_eff_net_test_fps"
    seq_len = 30
    model = create_neural_net(120, 160, 5e-5, 3, seq_len)
    model.summary()
    model.save(name)


if __name__ == "__main__":
    cnn = create_cnn_only(120, 160, 5e-5, 3)
    cnn.summary()



