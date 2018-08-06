from .accuracy import Accuracy
from .metric_bundle import MetricBundle


def create_metric_bundle(metrics):
    metric_bundle = MetricBundle([metric() for metric in metrics])
    return metric_bundle
