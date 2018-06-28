from .accuracy import Accuracy


class MetricFactory:
    @classmethod
    def create_metric(cls, metric_name):
        if metric_name == 'Accuracy':
            metric = Accuracy()
        else:
            raise KeyError('Undefined metric: ' + metric_name)
        return metric

    @classmethod
    def create_metric_bundle(cls, metric_names):
        metric_bundle = {}
        for metric_name in metric_names:
            metric_bundle[metric_name] = cls.create_metric(metric_name)
        return metric_bundle

