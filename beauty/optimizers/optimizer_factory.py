from torch import optim


class OptimizerFactory:
  @classmethod
  def create_optimizer(cls, name, parameters, **kwargs):
    if name == 'Adam':
        optimizer =  optim.Adam(
            parameters, lr=kwargs['lr'],
            betas=kwargs['betas'], weight_decay=kwargs['weight_decay']
        )
    elif name == 'SGD':
        optimizer =  optim.SGD(
            parameters, lr=kwargs['lr'],
            momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'],
            nesterov=True
        )
    else:
        raise KeyError('Unrecognized optimizer: ' + name)
    return optimizer
