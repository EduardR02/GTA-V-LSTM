import matplotlib.pyplot as plt
import config
import utils
import numpy as np
import time
import albumentations as A
import torch
import cv2


# Imagenet mean and std used in dinov2 training
ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD = (0.229, 0.224, 0.225)


def stack_and_convert(x, y, classifier_type, id2label):
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    y = torch.from_numpy(y)
    y_for_metrics = y
    if classifier_type == "seg_bce":
        # +1 because we already removed the background class in id2label
        y = torch.nn.functional.one_hot(y.long(), num_classes=len(id2label) + 1).permute(0, 3, 1, 2)
        # remove the class 0 (background), we have B, C, H, W , so remove the first one from C
        y = y[:, 1:]
    if classifier_type == "seg_cce":
        y = y.long()
    else:
        y = y.float()
    return x, y, y_for_metrics


def expand_imgs(imgs):
    # expand the images to 3 channels
    imgs = [np.stack([img, img, img], axis=-1) if len(img.shape) < 3 else img for img in imgs]
    return imgs


train_transform = A.Compose([
    A.PadIfNeeded(min_height=224, min_width=896, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=224, width=896),
    # A.Resize(width=448, height=448),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=224, min_width=896, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=224, width=896),
    # A.Resize(width=448, height=448),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])



def random_contrast():
    return RandomContrast(0.7)


def random_rotation():
    return RandomRotation(0.05, fill_mode="nearest")


def random_zoom():
    return RandomZoom(height_factor=(-0.05, -0.2))


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


def get_augmented_dataset(images, labels, shuffle=True):
    AUTOTUNE = tf.data.AUTOTUNE
    r_rotate = random_rotation()
    r_zoom = random_zoom()
    r_contrast = random_contrast()
    # force to allocate on CPU, as above tf 2.5 will allocate on GPU by default, and crash when data > 3GB
    # for some reason the preprocessing steps automatically get allocated to gpu, making them much slower because
    # of copying back and forth, so I just put one before everything to make sure
    with tf.device("CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        rng = tf.random.Generator.from_seed(time.time() % 1000, alg="philox")

    def seed_update_wrapper(img, lb):
        with tf.device("CPU:0"):
            seed = rng.make_seeds(2)[0]
            val = rng.uniform(shape=(2, 1))
            image, label = augment(img, lb, seed, val)
            return image, label

    def augment(image, label, seed, val):
        with tf.device("CPU:0"):
            # half the time, either rotate or zoom, and always augment brightness or contrast
            img = image
            """if val[0] <= 0.25:
                img = r_zoom(img)
            elif val[0] <= 0.5:
                img = r_rotate(img)"""
            if val[1] <= 0.5:
                img = tf.image.stateless_random_brightness(img, max_delta=0.3, seed=seed)
            else:
                img = r_contrast(img)
            return img, label

    with tf.device("CPU:0"):
        if shuffle:
            dataset = (dataset.shuffle(1000)
                       .map(seed_update_wrapper, num_parallel_calls=AUTOTUNE)
                       .batch(config.CNN_ONLY_BATCH_SIZE)
                       .shuffle(100)
                       .prefetch(AUTOTUNE)
                       )
        else:
            dataset = (dataset.map(seed_update_wrapper, num_parallel_calls=AUTOTUNE)
                       .batch(config.CNN_ONLY_BATCH_SIZE)
                       .prefetch(AUTOTUNE)
                       )
    return dataset


def test_augmentation():
    images, labels = utils.load_file(config.new_data_dir_name + config.data_name + "_0.h5")
    reverse_dict = {value: key for key, value in config.outputs.items()}
    dataset = get_augmented_dataset(images, labels)
    first_batch = None
    for batch in dataset:
        first_batch = batch
        break
    for i in range(30):
        random_idx = np.random.choice(labels.shape[0], 1, replace=False)[0]
        visualize(images[random_idx], first_batch[0][i])
        label_val = np.argmax(labels[random_idx])
        print("Label:", reverse_dict[label_val])


if __name__ == "__main__":
    test_augmentation()
