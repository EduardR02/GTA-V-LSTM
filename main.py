import random
import numpy as np
import mss
from grabscreen import grab_screen
import cv2
import time
from keys import PressKey, ReleaseKey, W, A, S, D
from training import height, width, color_channels, model_name, sequence_len
from tensorflow.keras.backend import clear_session
from tensorflow import compat
from tensorflow.keras.models import load_model
from collect_training_data import w, wa, wd, s, a, d, key_check, monitor
from lenet import create_neural_net

threshold_turn = 0
t_time = 0


def release_all():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def forward():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    if t_time != 0:
        time.sleep(t_time)
        ReleaseKey(D)


def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(D)
    if t_time != 0:
        time.sleep(t_time)
        ReleaseKey(A)


def backwards():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    if t_time != 0:
        time.sleep(t_time)
        ReleaseKey(A)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    if t_time != 0:
        time.sleep(t_time)
        ReleaseKey(D)


def output_key(x, y):
    prediction = [0, 0, 0, 0, 0, 0]
    prediction[x] = 1

    if y[x] > threshold_turn:
        if prediction == a:
            left()
            print("a", y[x])
        elif prediction == wa:
            forward_left()
            print("wa", y[x])
        elif prediction == wd:
            forward_right()
            print("wd", y[x])
        elif prediction == s:
            backwards()
            print("s", y[x])
        elif prediction == d:
            right()
            print("d", y[x])
        elif prediction == w:
            forward()
            print("w", y[x])
        else:
            print("error big time")
    else:
        forward()
        print("w", y[x])
    # so it doesn't get stuck
    # if random.randint(0, 5) == 3: PressKey(W)


# noinspection PyTypeChecker
def show_screen():
    sct = mss.mss()
    monitor = {'top': 27, 'left': 0, 'width': 800, 'height': 600}
    while 1:
        img = np.asarray(sct.grab(monitor))     # output is bgra
        # no need to convert color as cv2 displays bgr as rgb, capture is in bgr
        cv2.imshow('Car View RGB', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main_with_lstm():
    clear_session()
    config = compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    compat.v1.Session(config=config)
    t = time.time()
    net = load_model(model_name)
    print("Model took:", time.time() - t, "to load.")
    net.summary()
    sct = mss.mss()
    counter = 0
    t = time.time()
    images = []
    while 1:
        counter += 1
        # noinspection PyTypeChecker
        img = np.asarray(sct.grab(monitor))     # in bgra format
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # cv2 in comparison to keras uses width, height order
        img = cv2.resize(img, (width, height))
        img = img.reshape(-1, height, width, color_channels)

        if len(images) < sequence_len:
            images.append(img)
        else:
            # when sequence_len amount of images are captured predictions starts, length being kept at sequence_len
            images.pop(0)   # remove oldest image
            images.append(img)  # always have last seq_len images, put newest image at the end
            # create numpy array from list and reshape to correct dimensions, height, then width!!!!!!!!
            input_arr = np.concatenate(images, axis=0).reshape((-1, sequence_len, height, width, color_channels))
            # prediction shape is (1, predicted_class_one_hot)
            prediction_distribution = net.predict(input_arr)[0]
            # output array is of shape (1, 6), where 6 is the classes
            # and 6 the corresponding labels (like forward, backward)
            # argmax gives index of prediction
            prediction = np.argmax(prediction_distribution, axis=-1)  # get output of last image
            # press predicted keys and display prediction
            output_key(prediction, prediction_distribution)
        if (time.time() - t) >= 1:
            print("fps:", counter)
            counter = 0
            t = time.time()
        key = key_check()
        if "T" in key:
            release_all()
            break
        if "N" in key:
            release_all()
            images = []
            time.sleep(5)


if __name__ == "__main__":
    main_with_lstm()
    # show_screen()     # check alignment before running model
