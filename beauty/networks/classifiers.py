from torch import nn

from . import utils


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.linear = nn.Linear(input_channels, output_channels)
        self.softmax = nn.Softmax(dim=1)

        utils.init_modules(self.modules())

    def forward(self, input_):
        linear = self.linear(input_)
        softmax = self.softmax(linear)
        return softmax
