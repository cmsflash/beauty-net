from torch import nn


class _FeatureExtractor(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        self.feature_channels = feature_channels
