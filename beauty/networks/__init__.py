from torch import nn

from . import feature_extractors, classifiers
from .networks import BeautyNet


def create_model(model_config, device):
    feature_extractor = model_config.feature_extractor()
    classifier = model_config.classifier(
        feature_extractor.feature_channels, model_config.class_count
    )
    model = model_config.network(feature_extractor, classifier)
    model = nn.DataParallel(model).to(device)
    return model
