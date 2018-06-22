from .constant_lr import ConstantLr


def create_lr_scheduler(lr_config, optimizer):
    lr_scheduler = lr_config.lr_scheduler(optimizer, **vars(lr_config.config))
    return lr_scheduler
