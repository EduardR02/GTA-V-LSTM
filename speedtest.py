from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import InceptionV3
import numpy as np
import time
import config


def test_efficient_vs_inception():
    input_eff = 380
    input_incep = 299
    inception_v3 = InceptionV3(
        include_top=False,
        weights="imagenet",
        classes=config.output_classes,
        classifier_activation="softmax",
    )
    efficient_net_b4 = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        classes=config.output_classes,
        classifier_activation="softmax",
    )

    print("done building")

    for i in range(5):
        test_img_incep = np.random.uniform(-1, 1, size=(1, input_incep, input_incep, config.color_channels))
        t = time.time()
        inception_v3.predict(test_img_incep)
        print("inception prediction took:", time.time() - t)

    for i in range(5):
        test_img_eff = np.random.randint(255, size=(1, input_eff, input_eff, config.color_channels))
        t = time.time()
        efficient_net_b4.predict(test_img_eff)
        print("EFficientnet prediction took:", time.time() - t)


def predict_vs_other():
    input_eff = 380
    efficient_net_b4 = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        classes=config.output_classes,
        classifier_activation="softmax",
    )
    for i in range(20):
        test_img_eff = np.random.randint(255, size=(1, input_eff, input_eff, config.color_channels))
        t = time.time()
        print(efficient_net_b4.predict(test_img_eff).shape)
        print("EFficientnet prediction took:", time.time() - t)

    for i in range(20):
        test_img_eff = np.random.randint(255, size=(1, input_eff, input_eff, config.color_channels))
        t = time.time()
        print(efficient_net_b4(test_img_eff, training=False).numpy().shape)
        print("EFficientnet other method prediction took:", time.time() - t)


def test_output_shape():
    output_arr = np.random.uniform(-1, 1, size=(1, config.output_classes))
    print(output_arr)
    print(output_arr.shape)
    print("-----------------")
    # output_arr = np.random.uniform(-1, 1, size=(1, 10, 6))   # 10 is sequence length, 6 is number of classes
    t = time.time()
    print("concat took:", time.time() - t)
    x = output_arr[0]
    print(x)
    x = np.argmax(x, axis=-1)
    print(x.shape)
    print(x)


def test_list_creating(reps):
    t = time.time()
    # fastest by far
    for i in range(reps):
        new_list = [j for j in range(100)]
    print("First method took:", time.time() - t)
    t = time.time()
    for i in range(reps):
        new_list = []
        for j in range(100):
            new_list.append(j)
    print("Second method took:", time.time() - t)
    t = time.time()
    for i in range(reps):
        new_list = []
        for j in range(100):
            new_list += [j]
    print("Third method took:", time.time() - t)


def test_if_still_np():
    x = np.random.rand(100, 3, 4, 2)
    sequence = x[0:50:5]
    k = [0]*100
    k = k[0:50]
    print(len(k))
    print(sequence.shape)
    print(sequence)
    sequence = sequence[:-1]
    print(sequence.shape)


if __name__ == "__main__":
    test_if_still_np()
