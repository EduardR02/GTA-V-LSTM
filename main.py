from collections import deque

import numpy as np
import mss
import cv2
import time
import utils
from keys import PressKey, ReleaseKey, W, A, S, D
import config
from grabkeys import key_check
from threading import Thread as Worker
from training_new import load_model
from dataloader import val_transform
import torch

threshold_turn = 0
t_time = 0.05


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


def simple_output_key(prediction):
    press_keys = True
    output_dict = {"w": 0, "a": 1, "s": 2, "d": 3}
    prediction = prediction.cpu().detach().numpy()
    thresholds = np.array([0.5, 0.5, 0.5, 0.5])     # w a s d
    result = (prediction >= thresholds).astype(int).squeeze()
    if press_keys:
        if result[0] == 1:
            PressKey(W)
        else:
            ReleaseKey(W)
        if result[1] == 1:
            PressKey(A)
            t1 = Worker(target=sleep_and_release, args=(A,))
            t1.start()
        else:
            ReleaseKey(A)
        if result[2] == 1:
            PressKey(S)
        else:
            ReleaseKey(S)
        if result[3] == 1:
            PressKey(D)
            t2 = Worker(target=sleep_and_release, args=(D,))
            t2.start()
        else:
            ReleaseKey(D)
    if np.sum(result) == 0:
        print("nothing pressed, max val was:", prediction.max())
    else:
        # print which keys are pressed in one line
        print("".join([key for key, value in output_dict.items() if result[value] == 1]))


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
    model = load_model(sample_only=True)
    sct = mss.mss()
    max_t_time = 1.0 / utils.get_fps_ratio()
    counter = 0
    t = time.time()
    while True:
        counter += 1
        img = get_screencap_img(sct)
        img = val_transform(image=img)["image"]
        img = img[None, ...]
        img = img.pin_memory().to("cuda")
        prediction, _ = model(img)
        prediction = torch.nn.functional.sigmoid(prediction)
        simple_output_key(prediction)
        key = key_check()
        if (time.time() - t) >= 1:
            print("fps:", counter)
            counter = 0
            t = time.time()
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
    model = load_model(sample_only=True)
    sct = mss.mss()
    max_t_time = 1.0 / utils.get_fps_ratio()
    t = time.time()
    counter = 0
    # Use a deque to store timestamped images
    max_stored_images = 20  # Adjust this value based on your memory constraints
    image_buffer = deque(maxlen=max_stored_images)

    desired_interval = config.sequence_stride / config.fps_at_recording_time
    while True:
        counter += 1
        current_time = time.time()
        img = get_screencap_img(sct)
        img = val_transform(image=img)["image"]

        image_buffer.append((current_time, img))    # timestamp for selecting images with best stride
        selected_images = select_images(image_buffer, desired_interval, config.sequence_len)

        img_tensor = torch.stack(selected_images, dim=0)   # first timedim, then batch
        img_tensor = img_tensor[None, ...]
        img_tensor = img_tensor.pin_memory().to("cuda")
        prediction, _ = model(img_tensor)
        prediction = torch.nn.functional.sigmoid(prediction)
        simple_output_key(prediction)
        key = key_check()
        if (time.time() - t) >= 1:
            print("fps:", counter)
            counter = 0
            t = time.time()
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


def select_images(image_buffer, desired_interval, sequence_len):
    if len(image_buffer) < 2:
        return [img for _, img in image_buffer]

    current_time = image_buffer[-1][0]
    selected_images = [image_buffer[-1][1]]  # Start with the most recent image

    # Create a copy of the buffer to avoid modifying the original
    available_images = list(image_buffer)[:-1]  # Exclude the most recent image

    for i in range(1, sequence_len):
        if not available_images:
            break

        target_time = current_time - i * desired_interval
        best_image_index = min(range(len(available_images)),
                               key=lambda i: abs(available_images[i][0] - target_time))

        selected_images.append(available_images[best_image_index][1])

        # Remove the selected image and all newer images
        available_images = available_images[:best_image_index]

    # Pad with the oldest selected image if we don't have enough
    while len(selected_images) < sequence_len:
        selected_images.append(selected_images[-1])

    return list(reversed(selected_images))


def get_screencap_img(sct):
    img = np.asarray(sct.grab(config.monitor))  # in bgra format
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    # cv2 in comparison to keras uses width, height order
    img = cv2.resize(img, (config.width, config.height))
    # ! check again with new data
    img = img.reshape(config.height, config.width, config.color_channels)
    return img


if __name__ == "__main__":
    main_with_lstm()
    # main_with_cnn()
    # show_screen()     # check alignment before running model
