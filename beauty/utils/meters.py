from numbers import Number


class Meter:
    def __init__(self, label='Default', initial=None):
        self.label = label
        self.initial = initial
        self.value = None
        self.measure = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = self.initial or 0.
        self.measure = self.initial or 0.
        if self.initial is None:
            self.count = 0
        else:
            self.count = 1

    def update(self, value, multiplicity=1):
        self.value = value
        self.count += multiplicity

    def __str__(self):
        string = f'{self.label}: {self.value:5.3} ({self.measure:5.3})'
        return string


class AverageMeter(Meter):
    def update(self, value, multiplicity=1):
        super().update(value, multiplicity)
        self.measure = (
            (self.measure * (self.count - multiplicity) + value * multiplicity)
            / self.count
        )


class MaxMeter(Meter):
    def __init__(self, label='Default', initial=None):
        self.latest = None
        super().__init__(label, initial)

    def reset(self):
        super().reset()
        self.latest = False

    def update(self, value, multiplicity=1):
        super().update(value, multiplicity)
        if value > self.measure:
            self.measure = value
            self.latest = True
        else:
            self.latest = False

    def __str__(self):
        if self.latest:
            marker = '*'
        else:
            marker = ''
        string = f'{super().__str__()}{marker}'
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

    def __add__(self, other_bundle):
        assert self.meters.keys().isdisjoint(other_bundle.meters.keys())
        bundle = MeterBundle(self.meter.values() + other_bundle.meters.values())
        return bundle

    def __str__(self):
        string = '\t'.join([str(meter) for meter in self.meters.values()])
        return string


class ModelMeters:
    def __init__(self, metrics):
        self.time_meter = AverageMeter('Time')
        self.loss_meter = AverageMeter('Loss')
        self.metric_meters = metrics.create_average_meters()

    def reset(self):
        self.time_meter.reset()
        self.loss_meter.reset()
        self.metric_meters.reset()

    def update(self, time, loss, metric_bundle, batch_size=1):
        self.time_meter.update(time)
        self.loss_meter.update(loss.item(), batch_size)
        self.metric_meters.update(metric_bundle)

    def __str__(self):
        string = f'{self.time_meter}\t{self.loss_meter}\t{self.metric_meters}'
        return string
