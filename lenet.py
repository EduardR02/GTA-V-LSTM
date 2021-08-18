from tensorflow.keras.layers import Dense, LSTM, Input
import numpy as np
import time
from tensorflow.keras.applications import EfficientNetB3, InceptionV3
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, Flatten, Dropout
from tensorflow.keras.models import load_model


# height first, then width is keras order
def create_neural_net(height, width, lr, color_channels, sequence_length):
    np.random.seed(1000)

    inputs = Input(shape=(sequence_length, height, width, color_channels))
    cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(height, width, color_channels))
    cnn = Model(inputs=cnn.input, outputs=cnn.layers[-1].output)
    # cnn.trainable = False
    x = TimeDistributed(cnn)(inputs)
    x = TimeDistributed(Flatten())(x)

    x = LSTM(256, input_shape=(sequence_length, (height, width, color_channels)), return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)

    outputs = Dense(6, activation="softmax")(x)
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
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


