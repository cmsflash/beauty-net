import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset

from . import transforms


class ImageNet(Dataset):
    def __init__(self, data_config):
        super().__init__()
        self.transform = transforms.create_transform(data_config.transforms)
        self.data_list = self._read_data_list(
            data_config.data_dir, data_config.data_list_path
        )
        self.discrete = data_config.discrete

    @classmethod
    def _read_data_list(cls, data_dir, data_list_path):
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

        image = self.transform(image)
        if self.discrete:
            score = round(score) - 1
        score = torch.tensor(score)

        return index, image, score

    def _read_example(self, index):
        image_path, score = self.data_list[index]
        image = Image.open(image_path)
        return image, score
