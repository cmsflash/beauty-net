import numpy as np
import cv2
import random
import math


class StaticTransforms:
  @classmethod
  def create_transform(cls, name, new_size):
    if name == 'Indentity':
        transform = StaticIdentity()
    elif name == 'Resize':
        transform = StaticResize(new_size)
    elif name == 'Horizontal Flip':
        transform = StaticHorizontalFlip(0.5)
    elif name == 'Data Augment':
        transform = StaticDataAugment(new_size, 0.5)
    else:
      raise KeyError('Illegal resize_method: ' + name)
    return transform


class StaticIdentity:
    def __init__(self):
        pass

    def __call__(self, image):
        return image


class StaticResize:
    def __init__(self, new_size):
        self.new_size = tuple(reversed(new_size))

    def __call__(self, image):
        image = cv2.resize(image, self.new_size)
        image = image.astype(np.float32)
        return image


class StaticHorizontalFlip:
    def __init__(self, frequency):
        self.frequency = frequency
        random.seed()

    def __call__(self, image):
        random_score = random.random()
        if random_score <= self.frequency:
            image = np.flip(image, axis=1).copy()
        return image

class StaticDataAugment:
    def __init__(self, new_size, flip_frequency):
        self.resize = StaticResize(new_size)
        self.flip = StaticHorizontalFlip(flip_frequency)

    def __call__(self, image):
        image = self.resize(image)
        image = self.flip(image)
        return image
