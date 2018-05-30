import numpy as np

import torch
from torch import nn


class Accuracy(nn.Module):
    def forward(self, prediction, truth):
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        correct = prediction == truth
        accuracy = correct.float().mean()
        return accuracy
