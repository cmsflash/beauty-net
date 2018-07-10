from numbers import Number


class Meter:
    def __init__(self, label='Default', initial=None):
        self.label = label
        self.initial = initial
        self.reset()

    def reset(self):
        self.val = self.initial or 0.
        self.measure = self.initial or 0.
        if self.initial is None:
            self.count = 0
        else:
            self.count = 1


class AverageMeter(Meter):
    def update(self, val, n=1):
        self.val = val
        self.measure = self.measure * self.count + val * n / (self.count + n)
        self.count += n

    def __str__(self):
        string = '{} {:5.3} ({:5.3})'.format(self.label, self.val, self.measure)
        return string


class MaxMeter(Meter):
    def reset(self):
        super().reset()
        self.latest = False

    def update(self, val, n=1):
        self.val = val
        if val > self.measure:
            self.measure = val
            self.latest = True
        else:
            self.latest = False

    def __str__(self):
        if self.latest:
            marker = '*'
        else:
            marker = ''
        string = '{}: {:5.3} [{:5.3}]{}'.format(
            self.label, self.val, self.measure, marker
        )
        return string


class MeterBundle:
    def __init__(self, meters):
        self.meters = {meter.label: meter for meter in meters}

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, other_bundle):
        assert self.meters.keys() == other_bundle.meters.keys()
        for label in self.meters.keys():
            other_meter = other_bundle.meters[label]
            self.meters[label].update(other_meter.measure, other_meter.count)

    def __str__(self):
        string = '\t'.join([str(meter) for meter in self.meters.values()])
        return string
