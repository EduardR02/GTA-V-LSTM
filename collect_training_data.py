from collections import Counter
from random import shuffle
import numpy as np
import mss
import cv2
import time
import os
import pandas as pd
from grabkeys import key_check
from training import sequence_data
import h5py
from threading import Thread as Worker
import config

curr_file_index = 0
gb_per_file = 2

monitor = {'top': 27, 'left': 0, 'width': 800, 'height': 600}
w = [1, 0, 0, 0, 0, 0, 0]
a = [0, 1, 0, 0, 0, 0, 0]
s = [0, 0, 1, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0]
nothing = [0, 0, 0, 0, 0, 1, 0]


def convert_output(key):

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
        output = nothing

    return output


def start():
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Go!")


def main(only_fps_test_mode=False):
    global curr_file_index
    correct_file_counter(True)  # start with new file
    train_images = []
    train_labels = []
    # start()
    paused = False
    counter = 0
    counter2 = 0
    n = 1
    t = time.time()
    sct = mss.mss()
    key_check()
    start()
    while True:
        if not paused:
            counter += 1
            counter2 += 1
            img, label = get_image_and_label(sct)
            train_images.append(img)
            train_labels.append(label)

            if len(train_labels) % 1000 == 0:
                print(counter)
                print("Average Fps:", counter / n)
                if not only_fps_test_mode:
                    # reduces fps a little for the time while saving, but without would stop loop for time of exec,
                    # meaning gaps in the data, multiprocessing module works worse, halts execution for a short time.
                    t1 = Worker(target=save_with_hdf5, args=(train_images[:], train_labels[:]))
                    t1.start()
                    train_images = []
                    train_labels = []
            if time.time() - t >= 1:
                t = time.time()
                n += 1
                counter2 = 0
        else:
            time.sleep(0.1)
        key = key_check()
        if "T" in key:
            if paused:
                print("unpaused")
            else:
                # if you press pause it probably means you want to discard last x frames
                if len(train_labels) > 300 and not only_fps_test_mode:
                    train_images = train_images[:-300]
                    train_labels = train_labels[:-300]
                    save_with_hdf5(train_images, train_labels)
                print("paused")
                curr_file_index = curr_file_index + 1
            paused = not paused
            train_images = []
            train_labels = []
            time.sleep(1)
            t = time.time()


def get_image_and_label(sct):
    img = np.asarray(sct.grab(monitor))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Car View", img)
    img = cv2.resize(img, (config.width, config.height))
    key = key_check()
    output = convert_output(key)
    return img, np.asarray(output)


def correct_file_counter(called_on_start):
    global curr_file_index
    filename = config.new_data_dir_name + config.data_name + f"_{curr_file_index}.h5"
    # iterate  to writable file
    while os.path.isfile(filename) and (called_on_start or os.stat(filename).st_size > (1024 ** 3 * gb_per_file)):
        curr_file_index = curr_file_index + 1
        filename = config.new_data_dir_name + config.data_name + f"_{curr_file_index}.h5"


def save_with_hdf5(data_x, data_y):
    global curr_file_index
    correct_file_counter(False)
    filename = config.new_data_dir_name + config.data_name + f"_{curr_file_index}.h5"
    # both are lists of numpy array, retain structure but make it np array
    data_x = np.stack(data_x, axis=0)
    data_y = np.stack(data_y, axis=0)
    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("images", data=data_x, chunks=True,
                              maxshape=(None, config.height, config.width, config.color_channels))
            hf.create_dataset("labels", data=data_y, chunks=True,
                              maxshape=(None, config.output_classes))
    else:
        with h5py.File(filename, 'a') as hf:
            hf["images"].resize((hf["images"].shape[0] + data_x.shape[0]), axis=0)
            hf["images"][-data_x.shape[0]:] = data_x

            hf["labels"].resize((hf["images"].shape[0] + data_y.shape[0]), axis=0)
            hf["labels"][-data_y.shape[0]:] = data_y
    file_size = os.stat(filename).st_size
    print(file_size)
    if file_size > (1024 ** 3 * gb_per_file): curr_file_index += 1
    return


def display_data(data):
    df = pd.DataFrame(data)
    print(df.head())
    print(Counter(df[1].apply(str)))


def normalize(data):
    list_wa = []
    list_w = []
    list_wd = []
    list_s = []
    list_d = []
    list_a = []

    for i in data:
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


def show_training_data():
    # see what happens if you change height and width, don't mess up on reshaping for the neural net
    image_data_train = load_data()[0]
    image_data_train = np.stack(image_data_train, axis=0)
    print(image_data_train.shape)
    # image_data_train, labels = sequence_data(training_data, shuffle_bool=True, incorporate_fps=True)
    i = 0
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
    print("loading training data")
    images = []
    labels = []
    data_index = 0
    filename = config.new_data_dir_name + config.data_name + f"_{data_index}.h5"
    while os.path.isfile(filename):
        with h5py.File(filename, "r") as hf:
            print(hf.keys())
            labels += list(hf.get("labels"))
            images += list(hf.get("images"))
        data_index += 1
        filename = config.new_data_dir_name + config.data_name + f"_{data_index}.h5"
    print(f"done loading data, loaded {data_index} parts.")
    return images, labels


if __name__ == "__main__":
    # load_data()
    # normalize()
    main(False)
    # show_training_data()

