# alexnet.py

""" AlexNet.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np


def alexnet(height, width, lr, color_channels):
    np.random.seed(1000)

    # Instantiate an empty model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(height, width, color_channels), kernel_size=(11, 11), strides=(4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, activation="relu"))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation="relu"))
    # Add Dropout
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(6, activation="softmax"))

    model.summary()

    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
