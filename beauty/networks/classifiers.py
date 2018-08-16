from torch import nn
from torch.nn import functional as f

from . import weight_init


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)
        weight_init.init(self.modules())

    def forward(self, input_):
        linear = self.linear(input_)
        softmax = f.softmax(linear, dim=1)
        return softmax
