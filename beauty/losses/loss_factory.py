from torch import nn


class LossFactory:
    @classmethod
    def create_loss(cls, name):
        if name == 'L1':
            loss = nn.L1Loss()
        elif name == 'Cross Entropy':
            loss = nn.CrossEntropyLoss()
        else:
            raise KeyError('Undefined loss: ' + name)
        return loss
