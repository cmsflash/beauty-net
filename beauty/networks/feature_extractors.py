import math

import torch
import torch.nn as nn

from .submodules import *


class _FeatureExtractor(nn.Module):
    def get_feature_channels(self):
        raise NotImplementedError()


class MobileNetV2(_FeatureExtractor):
    def __init__(self, feature_channels=1280):
        super().__init__()
        self.interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        in_channels = 32
        self.feature_channels = feature_channels
        self.features = [conv(3, in_channels, stride=2)]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(in_channels, output_channel, s, t)
                    )
                else:
                    self.features.append(
                        InvertedResidual(in_channels, output_channel, 1, t)
                    )
                in_channels = output_channel
        self.features.append(conv(in_channels, self.feature_channels, 1))
        self.features = nn.Sequential(*self.features)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self._initialize_weights()

    def forward(self, input):
        feature_map = self.features(input)
        global_pool = self.global_pool(feature_map)
        feature = torch.squeeze(torch.squeeze(global_pool, dim=3), dim=2)
        return feature

    def get_feature_channels(self):
        return self.feature_channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
