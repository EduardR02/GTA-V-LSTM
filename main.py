import numpy as np
import mss
import cv2
import time
import utils
from keys import PressKey, ReleaseKey, W, A, S, D
import config
from tensorflow.keras.backend import clear_session
from tensorflow import compat
from tensorflow.keras.models import load_model
from lenet import inception_expose_feature_layer, lstm_only
from grabkeys import key_check
from threading import Thread as Worker

threshold_turn = 0
t_time = 0.04


def sleep_and_release(key):
    time.sleep(t_time)
    ReleaseKey(key)


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


def right(short_press):
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    if short_press:
        t1 = Worker(target=sleep_and_release, args=(D,))
        t1.start()


def left(short_press):
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(D)
    if short_press:
        t1 = Worker(target=sleep_and_release, args=(A,))
        t1.start()


def backwards():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left(short_press):
    PressKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    if short_press:
        t1 = Worker(target=sleep_and_release, args=(A,))
        t1.start()


def forward_right(short_press):
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    if short_press:
        t1 = Worker(target=sleep_and_release, args=(D,))
        t1.start()


def output_key(prediction):
    # index_list = [1, 3, 4, 5]
    max_idx = np.argmax(prediction)
    output = ""
    previous_output_size = 7
    short_press = False
    if max_idx >= previous_output_size:
        short_press = True
        if max_idx == previous_output_size:
            max_idx -= 6
        else:
            max_idx -= 5

    if prediction[max_idx] > threshold_turn:
        if max_idx == config.outputs["a"]:
            left(short_press)
            output = "a" if not short_press else "short a"
        elif max_idx == config.outputs["wa"]:
            forward_left(short_press)
            output = "wa" if not short_press else "short wa"
        elif max_idx == config.outputs["wd"]:
            forward_right(short_press)
            output = "wd" if not short_press else "short wd"
        elif max_idx == config.outputs["s"]:
            backwards()
            output = "s"
        elif max_idx == config.outputs["d"]:
            right(short_press)
            output = "d" if not short_press else "short d"
        elif max_idx == config.outputs["w"]:
            forward()
            output = "w"
        elif max_idx == config.outputs["nothing"]:
            release_all()
            output = "nothing"
        else:
            print("error big time")
    else:
        forward()
        output = "w"
    pred_value = prediction[max_idx] if not short_press else prediction[max_idx + 6] if max_idx == 1 else prediction[
        max_idx + 5]
    print("Prediction:", output, "Value", pred_value)


# noinspection PyTypeChecker
def show_screen():
    sct = mss.mss()
    monitor = config.monitor
    while 1:
        img = np.asarray(sct.grab(monitor))  # output is bgra
        """img = cv2.resize(img, (int(config.width), int(config.height)))
        img = cv2.resize(img, (800, 600))"""
        # img = get_screencap_img(sct)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # no need to convert color as cv2 displays bgr as rgb, capture is in bgr
        cv2.imshow(f'Car View {config.width}x{config.height}', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main_with_cnn():
    global t_time
    setup_tf()
    model = load_model(config.cnn_only_name)
    model.summary()
    sct = mss.mss()
    max_t_time = 1.0 / utils.get_fps_ratio()
    while True:
        img = get_screencap_img(sct)
        img = img.reshape((-1, config.height, config.width, config.color_channels))
        prediction = model.predict(img)[0]
        output_key(prediction)
        key = key_check()
        if "T" in key:
            release_all()
            break
        if "N" in key:
            release_all()
            time.sleep(5)
        if "X" in key:
            t_time = max(0.01, t_time - 0.01)
            print(t_time)
        if "B" in key:
            t_time = min(max_t_time, t_time + 0.01)
            print(t_time)


def main_with_lstm():
    global t_time
    setup_tf()
    t = time.time()
    feature_extractor = load_model(config.cnn_only_name)
    feature_extractor = inception_expose_feature_layer(feature_extractor)
    lstm_model = load_model(config.model_name)
    # lstm_model = lstm_only()
    print("Models took:", time.time() - t, "to load.")
    feature_extractor.summary()
    lstm_model.summary()
    sct = mss.mss()
    counter = 0
    t = time.time()
    max_t_time = round(1.0 / utils.get_fps_ratio(), 2)  # round to 2 digits
    images = []
    while 1:
        counter += 1
        img = get_screencap_img(sct)

        if len(images) < config.sequence_len:
            images.append(img)
        else:
            images.pop(0)  # remove oldest image
            images.append(img)  # always have last seq_len images, put newest image at the end
            # create numpy array from list and reshape to correct dimensions, height, then width!!!!!!!!
            input_arr = np.stack(images, axis=0)
            features = feature_extractor.predict(input_arr)
            # add single dimension at front for prediction
            features = features.reshape((-1, config.sequence_len, features.shape[-1]))
            prediction = lstm_model.predict(features)[0]
            # press predicted keys and display prediction
            output_key(prediction)
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
        if "X" in key:
            t_time = max(0.01, t_time - 0.01)
            print(t_time)
        if "B" in key:
            t_time = min(max_t_time, t_time + 0.01)
            print(t_time)


def get_screencap_img(sct):
    img = np.asarray(sct.grab(config.monitor))  # in bgra format
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    # cv2 in comparison to keras uses width, height order
    img = cv2.resize(img, (config.width, config.height))
    # ! check again with new data
    img = img.reshape(config.height, config.width, config.color_channels)
    return img


def setup_tf():
    clear_session()
    config_var = compat.v1.ConfigProto()
    config_var.gpu_options.allow_growth = True
    compat.v1.Session(config=config_var)


if __name__ == "__main__":
    # main_with_cnn()
    main_with_lstm()
    # show_screen()     # check alignment before running model
