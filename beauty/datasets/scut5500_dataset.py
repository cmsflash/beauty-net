import torch
import os.path as osp
import numpy as np
from scipy.misc import imread

from torch.utils.data import Dataset

from .static_transforms import StaticTransforms
from .utils import *


class Scut5500Dataset(Dataset):
    def __init__(self, data_config):
        super().__init__()
        self.input_size = data_config.input_size
        self.transform_method = data_config.transform_method
        self.data_list = self._read_data_list(
            data_config.data_dir, data_config.data_list_path
        )

    def _read_data_list(self, data_dir, data_list_path):
        data_list = []
        with open(data_list_path) as data_list_file:
            for line in data_list_file:
                image_name, score = line.split()
                image_path = osp.join(data_dir, image_name)
                score = float(score)
                data_list.append((image_path, score))
        return data_list

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
        score = round(score) - 1
        score = torch.tensor(score)

        return image, score

    def _normalize(self, image):
        image = (image - IMAGENET_MEANS) / RGB_MAX
        return image

    def _read_example(self, index):
        image_path, score = self.data_list[index]
        image = imread(image_path)
        return image, score
