from collections import Counter
from random import shuffle
import numpy as np
import mss
import cv2
import time
import os
import pandas as pd
from grabkeys import key_check
from training import width, height, sequence_data, color_channels
import gzip

file_name_1 = "training_data_rgb.npy"
file_name_2 = "training_data_rgb_part_2s.npy"
file_name_3 = "training_data_rgb_part_3.npy"
file_name_4 = "training_data_rgb_part_4.npy"
file_name_5 = "training_data_rgb_part_5.npy"
file_name_6 = "training_data_rgb_part_6.npy"
final_data_name = "training_data_for_lstm_rgb_full.npy"
file_name_7 = "training_data_rgb_part_7.npy"
file_name_8 = "training_data_rgb_part_8.npy"
file_name_9 = "training_data_rgb_part_9.npy"
file_name_10 = "training_data_rgb_get_back_on_road"
current_file_name = file_name_10
huge_dataset_filename = "training_data_full_lstm_unshuffled_less_balanced_1.npy"
training_data = []


w = [1, 0, 0, 0, 0, 0]
a = [0, 1, 0, 0, 0, 0]
s = [0, 0, 1, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0]
wa = [0, 0, 0, 0, 1, 0]
wd = [0, 0, 0, 0, 0, 1]


def convert_output(key):
    output = [0, 0, 0, 0]

    if "A" in key and "W" in key:
        output = wa
    elif "D" in key and "W" in key:
        output = wd
    elif "S" in key:
        output = s
    elif "D" in key:
        output = d
    elif "A" in key:
        output = a
    elif "W" in key:
        output = w
    else:
        output = w

    return output


def start():
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Go!")


# noinspection PyTypeChecker
def main(only_fps_test_mode=False):     # check training data on start of file before running
    global training_data
    # start()
    paused = True
    print("starting paused")
    counter = 0
    counter2 = 0
    n = 1
    t = time.time()
    sct = mss.mss()
    monitor = {'top': 27, 'left': 0, 'width': 800, 'height': 600}
    while True:
        if not paused:
            if only_fps_test_mode:
                counter += 1
                counter2 += 1
            img = np.asarray(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow("Car View", img)
            img = cv2.resize(img, (width, height))
            key = key_check()
            output = convert_output(key)
            training_data.append([img, output])
            """if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break"""

            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(current_file_name, training_data)
            if only_fps_test_mode:
                if time.time() - t >= 1:
                    print("Current Fps:", counter2, end="; ")
                    print("Average Fps:", counter / n)
                    t = time.time()
                    n += 1
                    counter2 = 0

        key = key_check()
        if "T" in key:
            if paused:
                paused = False
                print("unpaused immediately")
                # time.sleep(1)
                # start()
            else:
                # if you press pause it probably means you want to discard last x frames
                paused = True
                # training_data = training_data[:-300]
                print("paused")
                time.sleep(1)


def display_data():
    df = pd.DataFrame(training_data)
    print(df.head())
    print(Counter(df[1].apply(str)))


def normalize():
    list_wa = []
    list_w = []
    list_wd = []
    list_s = []
    list_d = []
    list_a = []

    for i in training_data:
        img = i[0]
        dd = i[1]

        if dd == w:
            list_w.append([img, dd])
        elif dd == wa:
            list_wa.append([img, dd])
        elif dd == wd:
            list_wd.append([img, dd])
        elif dd == a:
            list_a.append([img, dd])
        elif dd == s:
            list_s.append([img, dd])
        elif dd == d:
            list_d.append([img, dd])
        else:
            print("uh oh stinky error")
    list_w = list_w[:len(list_w)//2]
    list_wa = list_wa[:len(list_wa)][:len(list_wd)]
    list_wd = list_wd[:len(list_wa)][:len(list_wd)]
    list_s = list_s[:len(list_w)]
    list_a = list_a[:len(list_a)][:len(list_d)]
    list_d = list_d[:len(list_a)][:len(list_d)]

    print("list_w:", len(list_w))
    print("list_wa:", len(list_wa))
    print("list_wd:", len(list_wd))
    print("list_s:", len(list_s))
    print("list_d:", len(list_d))
    print("list_a:", len(list_a))

    all_data = list_w + list_a + list_wa + list_wd + list_d + list_s
    print("all_data:", len(all_data))
    t = time.time()
    print(time.time() - t)
    shuffle(all_data)
    print(time.time() - t)
    # np.save(final_data_name, all_data)
    print(time.time() - t)
    return all_data


def normalize_lstm():
    global training_data
    display_data()
    list_res = []
    last_skipped = 0
    for i in training_data:
        img = i[0]
        dd = i[1]
        last_label = w if len(list_res) == 0 else list_res[-1][1]
        if dd == w:
            if last_skipped > 2 and last_label == w:
                list_res.append([img, dd])
                last_skipped = 0
            elif last_label == w:
                last_skipped += 1
            else:
                list_res.append([img, dd])
                last_skipped = 0
        elif dd == wa:
            list_res.append([img, dd])
        elif dd == wd:
            list_res.append([img, dd])
        elif dd == a:
            list_res.append([img, dd])
        elif dd == s:
            list_res.append([img, dd])
        elif dd == d:
            list_res.append([img, dd])
        else:
            print("uh oh stinky error")
        if dd != w: last_skipped = 0
    training_data = list_res
    display_data()


def show_training_data():
    print(len(training_data))
    # see what happens if you change height and width, don't mess up on reshaping for the neural net
    # was testing to see what happens when width and height are swapped
    # image_data_train = np.array([i[0] for i in training_data]).reshape((-1, height, width, color_channels))
    image_data_train, labels = sequence_data(training_data, shuffle_bool=True, incorporate_fps=True)
    i = 0
    image_data_train = image_data_train.reshape((-1, height, width, color_channels))
    print(image_data_train.shape)
    t = time.time()
    count_frames = 0
    while True:
        img = image_data_train[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # because cv2 imshow uses bgr not rgb
        cv2.imshow("poop", img)
        if time.time() - t > 1:
            t = time.time()
            print("fps:", count_frames)
            count_frames = 0
        count_frames += 1
        i += 1
        if len(image_data_train) <= i:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def load_data():
    global training_data
    if os.path.isfile(current_file_name):
        print("loading training data")
        temp1 = list(np.load(file_name_1, allow_pickle=True))
        temp2 = list(np.load(file_name_2, allow_pickle=True))
        temp3 = list(np.load(file_name_3, allow_pickle=True))
        temp4 = list(np.load(file_name_4, allow_pickle=True))
        temp5 = list(np.load(file_name_5, allow_pickle=True))
        temp6 = list(np.load(file_name_6, allow_pickle=True))
        temp7 = list(np.load(file_name_7, allow_pickle=True))
        temp8 = list(np.load(file_name_8, allow_pickle=True))
        temp9 = list(np.load(file_name_9, allow_pickle=True))
        training_data = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8 + temp9
        print("done loading data")
    else:
        print("No training data found, starting from scratch")
        training_data = []


def load_for_show():
    global training_data
    training_data = np.load(file_name_9, allow_pickle=True)
    # training_data = training_data[:len(training_data) // 50]


def normalize_for_lstm_and_save():
    t = time.time()
    load_data()
    print("Lading data took:", time.time() - t, "seconds.")
    # normalize_lstm()
    print(len(training_data))
    t = time.time()
    np.save(final_data_name, training_data)
    print("Saving data took:", time.time() - t, "seconds.")


def compress_numpy_arr():
    load_for_show()
    f = gzip.GzipFile("compressed_full_dataset.npy.gz", "w")
    np.save(file=f, arr=training_data)
    f.close()


if __name__ == "__main__":
    current_file_name = file_name_10
    # normalize_for_lstm_and_save()
    # compress_numpy_arr()
    # load_for_show()
    # display_data()
    # show_training_data()

    # normalize()
    # main(False)
    # show_training_data()

