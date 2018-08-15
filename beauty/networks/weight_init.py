from torch import nn


def init(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu'
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            n = module.weight.size(1)
            nn.init.normal_(0, 0.01)
            nn.init.constant_(module.bias, 0)
