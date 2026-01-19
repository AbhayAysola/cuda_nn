import numpy as np
import struct

import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, labels_file, images_file, transform=None, target_transform=None):
        with open(images_file,'rb') as f:
            _, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.images = data.reshape((size, 1, nrows, ncols)) / 255
            self.images = self.images.astype("float32")
            self.images = torch.from_numpy(self.images)
        with open(labels_file,'rb') as f:
            _, size = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.labels = data.reshape(size)
            self.labels = torch.from_numpy(self.labels)

        self.transform = transform
        self.target_transform = target_transform
        pass 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


