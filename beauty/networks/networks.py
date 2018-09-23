import torch
from torch import nn


class BeautyNet(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, input_):
        feature = self.feature_extractor(input_)
        feature_vector = torch.squeeze(torch.squeeze(feature, dim=3), dim=2)
        classification = self.classifier(feature_vector)
        return classification
