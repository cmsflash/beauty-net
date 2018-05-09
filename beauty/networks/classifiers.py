from torch import nn

from .utils import *


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.linear = nn.Linear(input_channels, output_channels)
        self.softmax = nn.Softmax(dim=1)
        
        init_modules(self.modules())

    def forward(self, input):
        linear = self.linear(input)
        softmax = self.softmax(linear)
        return softmax
