from torch.optim.lr_scheduler import LambdaLR


class ConstantLr(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, lambda x: 1, last_epoch)
