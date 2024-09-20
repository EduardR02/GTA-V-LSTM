import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import bisect
import cv2
from matplotlib import pyplot as plt


ADE_MEAN = (0.485, 0.456, 0.406)
ADE_STD = (0.229, 0.224, 0.225)
height = 448
width = 448


class H5Dataset(Dataset):
    def __init__(self, data_dirs, train_split, is_train):
        self.train_split = train_split
        self.is_train = is_train
        self.transform = train_transform if is_train else val_transform
        self.data_files = [os.path.join(data_dir, f) for data_dir in data_dirs for f in sorted(os.listdir(data_dir))if f.endswith('.h5')]
        self.lookup_table = self._create_lookup_table()

    def _create_lookup_table(self):
        lookup = []
        total_samples = 0
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as f:
                total_samples += f['labels'].shape[0]
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

        with h5py.File(file_path, 'r') as f:
            data = f['images'][local_idx]
            label = f['labels'][local_idx].astype(np.int8).flatten()
            label = self._to_wasd(label)

        if self.transform:
            augmented = self.transform(image=data)
            data = augmented['image']

        return data, label

    def _get_split_idx(self):
        # exclusive for train, first index of val
        return int(self.lookup_table[-1] * self.train_split)

    def _to_wasd(self, label):
        new_label = np.zeros(4, dtype=np.float32)

        new_label[0] = label[0] | label[4] | label[5]  # w
        new_label[1] = label[1] | label[4]  # a
        new_label[2] = label[2]  # s
        new_label[3] = label[3] | label[5]  # d

        return new_label


train_transform = A.Compose([
    A.LongestMaxSize(max_size=max(height, width)),  # Resize the longest side to match the input size
    A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the smaller side
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ToTensorV2(),
])


val_transform = A.Compose([
    A.LongestMaxSize(max_size=max(height, width)),  # Resize the longest side to match the input size
    A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad the smaller side
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ToTensorV2(),
])


def get_dataloader(data_dir, batch_size, train_split, is_train, shuffle=True):
    dataset = H5Dataset(data_dir, train_split, is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    return dataloader


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
def invNormalize(x):
    return (x * np.array(ADE_STD)[None, None, :]) + np.array(ADE_MEAN)[None, None, :]


if __name__ == '__main__':
    data_dirs = ['data/turns', 'data/new_data']
    train_loader = get_dataloader(data_dirs, 32, 0.95, is_train=True)
    # vizualize data with matplotlib until stopped
    for data, label in train_loader:
        for i in range(data.size(0)):
            img = invNormalize(data[i].permute(1, 2, 0).numpy())
            print(label[i])
            plt.imshow(img)
            plt.show()
