from torch import nn

from . import feature_extractors, classifiers
from .networks import BeautyNet


_CLASS_COUNT = 5


def create_model(model_config, device):
    feature_extractor = model_config.feature_extractor()
    classifier = model_config.classifier(
        feature_extractor.feature_channels, _CLASS_COUNT
    )
    model = model_config.network(feature_extractor, classifier)
    model = nn.DataParallel(model).to(device)
    return model
