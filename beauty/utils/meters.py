class AverageMeter:
    def __init__(self, label='Default'):
        self.label = label
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MaxMeter:
    def __init__(self, label='Default'):
        self.label = label
        self.reset()

    def reset(self):
        self.val = 0.
        self.max = 0.
        self.latest = False
    
    def update(self, val, n=1):
        self.val = val
        if val > self.max:
            self.max = val
            self.latest = True
        else:
            self.latest = False

    def __str__(self):
        string = '{}: {:5.3}\tbest: {:5.3}{}\t'.format(
            self.label, self.val, self.max, ' *' if self.latest else ''
        )
        return string

 
class MeterBundle:
    def __init__(self, meters):
        self.meters = meters

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, other_bundle):
        assert self.meters.keys() == other_bundle.meters.keys()
        for label in self.meters.keys():
            self.meters[label].update(
                other_bundle.meters[label].avg, other_bundle.meters[label].count
            )
