import math

from torch import nn


def get_perfect_padding(kernel_size, dilation=1):
    padding = (kernel_size - 1) * dilation // 2
    return padding


def sequential(*modules):
    '''
    Returns an nn.Sequential object using modules with None's filtered
    '''
    modules = [module for module in modules if module is not None]
    return nn.Sequential(*modules)


def activation():
    activation = nn.ReLU6(inplace=True)
    return activation


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        self.stride = stride
        self.is_residual = self.stride == 1 and in_channels == out_channels
        channels = in_channels * expansion

        self.conv = sequential(
            conv(in_channels, channels, 1, activation=activation()),
            conv(
                channels, channels, 3, self.stride, groups=channels,
                activation=activation()
            ),
            conv(channels, out_channels, 1, activation=None)
        )

    def forward(self, x):
        conv = self.conv(x)
        if self.is_residual:
            output = conv + x
        else:
            output = conv
        return output


def inverted_residuals(
        in_channels, out_channels, expansion=6, stride=1, blocks=1
    ):
    residual_list = [
        InvertedResidual(in_channels, out_channels, expansion, stride)
    ] + [
        InvertedResidual(out_channels, out_channels, expansion, 1)
        for _ in range(blocks - 1)
    ]
    residuals = sequential(*residual_list)
    return residuals


def conv(
        in_channels, out_channels, kernel_size=3,
        stride=1, padding=None, dilation=1, groups=1,
        norm=nn.BatchNorm2d, activation=activation()
    ):
    padding = padding or get_perfect_padding(kernel_size, dilation)
    layer = sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False
        ),
        norm(out_channels),
        activation
    )
    return layer
