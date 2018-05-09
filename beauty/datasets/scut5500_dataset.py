import torch
import os.path as osp
import numpy as np
from scipy.misc import imread

from torch.utils.data import Dataset

from .static_transforms import StaticTransforms
from .utils import *


class Scut5500Dataset(Dataset):
    def __init__(self, data_dir, data_list, input_size, transform_method):
        super().__init__()
        self.input_size = input_size
        self.transform_method = transform_method
        self.data_list = []
        for line in data_list:
            image_name, score = line.split()
            image_path = osp.join(data_dir, image_name)
            score = float(score)
            self.data_list.append((image_path, score))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image, score = self._read_example(index)
        image_size = image.shape[:2]
        input_size = self.input_size
        transform = StaticTransforms.create_transform(
            self.transform_method, input_size
        )

        image = self._normalize(image)
        image = transform(image)
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        score = round(score)
        score = torch.tensor(score)

        return image, score

    def _normalize(self, image):
        image = (image - IMAGENET_MEANS) / RGB_MAX
        return image

    def _read_example(self, index):
        image_path, score = self.data_list[index]
        image = imread(image_path)
        return image, score
