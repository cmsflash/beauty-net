from torch import nn

from ..utils import meters


class MetricBundle(nn.Module):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = {metric.label: metric for metric in metrics}

    def forward(self, prediction, truth):
        metric_values = meters.MeterBundle([
            meters.Meter(label, metric(prediction, truth))
            for label, metric in self.metrics.items()
        ])
        return metric_values

    def create_max_meters(self):
        metric_values = self._create_meters(meters.MaxMeter)
        return metric_values

    def create_average_meters(self):
        metric_values = self._create_meters(meters.AverageMeter)
        return metric_values

    def _create_meters(self, meter_type):
        metric_values = meters.MeterBundle([
            meter_type(label, 0) for label, metric in self.metrics.items()
        ])
        return metric_values
