from torch import nn

from ..utils import meters


class MetricBundle:
    def __init__(self, metrics):
        self.metrics = {metric.label: metric for metric in metrics}

    def get_metrics(self, input):
        metric_values = meters.MeterBundle({
            meters.Meter(metric.label, metric(input))
            for label, metric in self.metrics.items()
        })
        return metric_values

    def create_best_meters(self):
        best_meters = meter
