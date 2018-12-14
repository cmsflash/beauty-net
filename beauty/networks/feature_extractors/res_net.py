from torchvision.models import resnet

from ._feature_extractor import _FeatureExtractor
from .. import submodules, weight_init


class ResNet50(_FeatureExtractor):

    def __init__(self, feature_channels=2048):
        super().__init__(feature_channels)

        self.res_net = resnet.resnet50()
        self.res_net.avgpool = submodules.identity()
        self.res_net.fc = submodules.identity()

        weight_init.init(self.modules())

    def forward(self, input_):
        
        n, _, h, w = input_.size()
        vectorized_output = self.res_net(input_)
        output = vectorized_output.view(n, 2048, h // 32, w // 32)
        return output

