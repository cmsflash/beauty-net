from torch import nn

from ..utils import meters


class MetricBundle:
    def __init__(self, metrics):
        self.metrics = {metric.label: metric for metric in metrics}

    def create_max_meters(self):
        metric_values = meters.MeterBundle([
            meters.MaxMeter(label, 0)
            for label, metric in self.metrics.items()
        ])
        return metric_values

    def create_average_meters(self):
        metric_values = meters.MeterBundle([
            meters.AverageMeter(label, 0)
            for label, metric in self.metrics.items()
        ])
        return metric_values

    def __call__(self, input, target):
        metric_values = meters.MeterBundle([
            meters.Meter(label, metric(input, target))
            for label, metric in self.metrics.items()
        ])
        return metric_values

