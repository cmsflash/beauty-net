from torch import nn
from torch.nn import init


def init_modules(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d):
            if module.bias is not None:
                init.uniform(module.bias)
            init.xavier_uniform(module.weight)
        if isinstance(module, nn.ConvTranspose2d):
            if module.bias is not None:
                init.uniform(module.bias)
            init.xavier_uniform(module.weight)
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                init.uniform(module.bias)
            init.xavier_uniform(module.weight)
