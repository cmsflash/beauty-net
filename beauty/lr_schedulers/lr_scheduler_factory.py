from torch.optim.lr_scheduler import *


class LrSchedulerFactory:
    @classmethod
    def create_lr_scheduler(cls, name, optimizer, start_epoch=0, **kwargs):
        last_epoch = start_epoch - 1
        if name == 'Constant':
            scheduler = LambdaLR(optimizer, lambda x: 1, last_epoch)
        elif name == 'Step':
            scheduler = StepLR(optimizer, kwargs['lr_step_size'], kwargs['gamma'], last_epoch)
        else:
            raise KeyError('Unrecognized learning rate scheduler: ' + name)
        return scheduler
