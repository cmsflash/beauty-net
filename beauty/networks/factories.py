from .beauty_net import BeautyNet
from .classifiers import SoftmaxClassifier
from .feature_extractors import MobileNetV2


class NetworkFactory:
    @classmethod
    def create_network(cls, name, feature_extractor, classifier):
        if name == 'BeautyNet':
            network = BeautyNet(feature_extractor, classifier)
        else:
           raise KeyError('Unrecognized network architecture: ' + name)
        return network
        

class ClassifierFactory:
    class_count = 5

    @classmethod
    def create_classifier(cls, name, feature_channels):
        if name == 'Softmax':
            classifier = SoftmaxClassifier(
                feature_channels, cls.class_count
            )
        else:
            raise KeyError('Unrecognize classifier: ' + name)
        return classifier
        

class FeatureExtractorFactory:
    @classmethod
    def create_feature_extractor(cls, name):
        if name == 'MobileNetV2':
            network = MobileNetV2()
        else:
           raise KeyError('Unrecognized feature extractor: ' + name)
        return network
       
