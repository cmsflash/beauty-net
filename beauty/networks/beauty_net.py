import torch
import torch.nn as nn

import math
import numpy as np


class BeautyNet(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, input):
        feature_vector = self.feature_extractor(input)
        classification = self.classifier(feature_vector)
        return classification
