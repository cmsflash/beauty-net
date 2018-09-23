import torch
from torch import nn
from torch.nn import functional as f


class BeautyNet(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, input_):
        feature = self.feature_extractor(input_)
        global_pool = f.adaptive_avg_pool2d(feature, 1)
        feature_vector = torch.squeeze(torch.squeeze(global_pool, dim=3), dim=2)
        classification = self.classifier(feature_vector)
        return classification
