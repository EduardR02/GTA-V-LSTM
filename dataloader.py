import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import stack, uint8
import h5py
import os
import bisect
import cv2
from matplotlib import pyplot as plt
import random
import config


ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD = (0.229, 0.224, 0.225)
height = 182
width = 252


class H5Dataset(Dataset):
    def __init__(self, data_dirs, train_split, is_train, classifier_type, flip_prob):
        self.train_split = train_split
        self.is_train = is_train
        self.classifier_type = classifier_type
        self.flip_prob = flip_prob
        self.transform = train_transform if is_train else val_transform
        self.data_files = [os.path.join(data_dir, f) for data_dir in data_dirs for f in sorted(os.listdir(data_dir))if f.endswith('.h5')]
        self.lookup_table = self._create_lookup_table()

    def _create_lookup_table(self):
        lookup = []
        total_samples = 0
        seq_len = 1 if not hasattr(self, 'sequence_len') else self.sequence_len
        seq_stride = 0 if not hasattr(self, 'sequence_stride') else self.sequence_stride
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as f:
                file_samples = f['labels'].shape[0]
                if seq_len > 1:
                    file_samples -= (seq_len - 1) * seq_stride
                if "stuck" in file_path:
                    # this is a bit confusing, but we only have to account for the shorter than amt_remove_after_pause
                    # case (otherwise -0), because we handle the case if it's longer with the sequence_len > 1 case
                    file_samples -= max(config.amt_remove_after_pause - ((seq_len - 1) * seq_stride), 0)
                if file_samples <= 0:
                    print(f"Skipping and removing {file_path} as it has {file_samples} samples (no valid samples)")
                    self.data_files.remove(file_path)
                    continue
                total_samples += file_samples
            lookup.append(total_samples)
        return lookup

    def __len__(self):
        train_split_len = self._get_split_idx()
        total_samples = self.lookup_table[-1]
        if self.is_train:
            return train_split_len
        else:
            return total_samples - train_split_len

    def __getitem__(self, idx):
        if not self.is_train:
            idx += self._get_split_idx()

        file_idx = bisect.bisect_right(self.lookup_table, idx)
        local_idx = idx - (self.lookup_table[file_idx - 1] if file_idx > 0 else 0)

        file_path = self.data_files[file_idx]

        if "stuck" in file_path:
            local_idx += config.amt_remove_after_pause

        with h5py.File(file_path, 'r') as f:
            image = f['images'][local_idx]
            label = f['labels'][local_idx]
            if self.classifier_type == "bce":
                label = label.astype(np.int8).flatten()
                label = self._to_wasd(label)
            else:
                label = label.flatten().astype(np.float32)
        if self.is_train and random.random() < self.flip_prob:
            image, label = self.flip_samples(image, label)
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def _get_split_idx(self):
        # exclusive for train, first index of val
        return int(self.lookup_table[-1] * self.train_split)

    def _to_wasd(self, label):
        # Ensure label is at least 2D
        if label.ndim == 1:
            label = label[np.newaxis, :]

        # Create a new array with the same shape as label, but with 4 columns instead of 7
        new_label = np.zeros((*label.shape[:-1], 4), dtype=np.float32)

        # Use broadcasting to apply the conversion to all sequences at once
        new_label[..., 0] = label[..., 0] | label[..., 4] | label[..., 5]  # w
        new_label[..., 1] = label[..., 1] | label[..., 4]  # a
        new_label[..., 2] = label[..., 2]  # s
        new_label[..., 3] = label[..., 3] | label[..., 5]  # d

        return new_label.squeeze()

    def flip_samples(self, images, labels):
        if labels.ndim == 1:
            augmented = flip_transform(image=images)['image']
            labels = labels[np.newaxis, :]
        else:
            augmented = np.stack([flip_transform(image=images[i])['image'] for i in range(images.shape[0])], axis=0)
        label = self.flip_labels(labels)
        return augmented, label.squeeze()

    def flip_labels(self, labels):
        labels[:, [1, 3]] = labels[:, [3, 1]]
        # additionally swap wd and wa
        if self.classifier_type == "cce":
            labels[:, [4, 5]] = labels[:, [5, 4]]
        return labels


class TimeSeriesDataset(H5Dataset):
    def __init__(self, data_dirs, train_split, is_train, classifier_type, flip_prob, sequence_len, sequence_stride):
        self.sequence_len = sequence_len
        self.sequence_stride = sequence_stride
        super().__init__(data_dirs, train_split, is_train, classifier_type, flip_prob)

    def __getitem__(self, idx):
        if not self.is_train:
            idx += self._get_split_idx()

        file_idx = bisect.bisect_right(self.lookup_table, idx)
        local_idx = idx - (self.lookup_table[file_idx - 1] if file_idx > 0 else 0)

        file_path = self.data_files[file_idx]
        total_valid_frames = self.lookup_table[file_idx] - (self.lookup_table[file_idx - 1] if file_idx > 0 else 0)
        sequence_range = (self.sequence_len - 1) * self.sequence_stride

        if local_idx + sequence_range >= total_valid_frames:
            # Adjust local_idx to get a valid sequence
            local_idx = total_valid_frames - sequence_range
        if "stuck" in file_path:
            local_idx += max(config.amt_remove_after_pause - sequence_range, 0)
        with h5py.File(file_path, 'r') as f:
            # Get sequence with stride
            indices = range(local_idx, local_idx + self.sequence_len * self.sequence_stride, self.sequence_stride)
            images = f['images'][indices]
            labels = f['labels'][indices]
        if self.classifier_type == "bce":
            labels = self._to_wasd(labels.astype(np.int8))
        else:
            labels = labels.astype(np.float32)
        if self.is_train and random.random() < self.flip_prob:
            images, labels = self.flip_samples(images, labels)
        if self.transform:
            images = stack([self.transform(image=images[i])['image'] for i in range(images.shape[0])], dim=0)
        return images, labels


flip_transform = A.Compose([
    A.HorizontalFlip(p=1),
])


train_transform = A.Compose([
    # A.LongestMaxSize(max_size=max(height, width)),  # Resize the longest side to match the input size
    A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the smaller side
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD, max_pixel_value=255.0),
    ToTensorV2(),
])


val_transform = A.Compose([
    # A.LongestMaxSize(max_size=max(height, width)),  # Resize the longest side to match the input size
    A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the smaller side
    A.Normalize(mean=ADE_MEAN, std=ADE_STD, max_pixel_value=255.0),
    ToTensorV2(),
])


def get_dataloader(data_dir, batch_size, train_split, is_train, classifier_type, sequence_len=1, sequence_stride=1, flip_prob=0., shuffle=True):
    if sequence_len > 1:
        dataset = TimeSeriesDataset(data_dir, train_split, is_train, classifier_type, flip_prob, sequence_len, sequence_stride)
    else:
        dataset = H5Dataset(data_dir, train_split, is_train, classifier_type, flip_prob)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        pin_memory_device="cuda",
        drop_last=True
    )
    return dataloader


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
def invNormalize(x):
    return (x * np.array(ADE_STD)[None, None, :]) + np.array(ADE_MEAN)[None, None, :]


if __name__ == '__main__':
    data_dirs = ['data/stuck']
    sequence_len = 2
    train_loader = get_dataloader(data_dirs, 1024, 0.95, True, "bce", sequence_len, 40, 0.5)
    # vizualize data with matplotlib until stopped
    for data, label in train_loader:
        for i in range(data.shape[0]):
            print(data.shape)
            if sequence_len > 1:
                for j in range(data.shape[1]):
                    print(data[i][j].max(), data[i][j].min())
                    img = invNormalize(data[i][j].permute(1, 2, 0).numpy())
                    print(label[i][j])
                    plt.imshow(img)
                    plt.show()
            else:
                print(data[i].max(), data[i].min())
                img = invNormalize(data[i].permute(1, 2, 0).numpy())
                print(label[i])
                plt.imshow(img)
                plt.show()

