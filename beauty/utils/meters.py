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

    def update(self, value_dict):
        other_bundle = self.from_dict(value_dict)
        assert self.meters.keys() == other_bundle.meters.keys()
        for label in self.meters.keys():
            other_meter = other_bundle.meter[label]
            self.meters[label].update(other_meter.measure, other_meter.count)

    @classmethod
    def from_dict(cls, value_dict):
        bundle = MeterBundle({
            label: Meter(value) for label, value in value_dict.items()
        })
        return bundle

    @classmethod
    def ensure_bundle(cls, maybe_bundle):
        if isinstance(maybe_bundle, MeterBundle):
            bundle = maybe_bundle
        elif isinstance(maybe_bundle, dict):
            bundle = cls.from_dict(maybe_bundle)
        else:
            raise TypeError('maybe_bundle must be a MeterBundle or a dict.')

    def __str__(self):
        string = '\t'.join(self.meters.values())
