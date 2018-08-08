import torch
import torch.nn as nn


class BeautyNet(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, input_):
        feature_vector = self.feature_extractor(input_)
        classification = self.classifier(feature_vector)
        return classification
