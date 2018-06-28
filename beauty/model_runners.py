import time

from torch.autograd import Variable

from .utils import meters


class Runner:
    def __init__(self, model, loss, metrics, input_config):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.input_config = input_config

        self.batch_time_meter = meters.AverageMeter()
        self.data_time_meter = meters.AverageMeter()
        self.loss_meter = meters.AverageMeter()
        self.metric_meters = meters.MeterBundle({
            metric_label: meters.AverageMeter(metric_label)
            for metric_label in metrics.keys()
        })

    def run(self, data_loader, epoch, **kwargs):
        self._set_model_mode()
        self._reset_stats()

        start_time = time.time()
        for i, inputs in enumerate(data_loader):
            data_time = time.time() - start_time

            inputs, targets = self._parse_data(inputs)
            loss, metric_bundle = self._forward(inputs, targets)

            self._step(loss, kwargs)

            batch_time = time.time() - start_time
            self._update_stats(batch_time, data_time, loss, metric_bundle)
            self.print_stats(epoch, i + 1, len(data_loader))
            start_time = time.time()

        return self.metric_meters

    def _set_model_mode(self):
        raise NotImplementedError()

    def _reset_stats(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.metric_meters.reset()

    def _update_stats(self, batch_time, data_time, loss, metric_bundle):
        self.batch_time_meter.update(batch_time)
        self.data_time_meter.update(data_time)
        self.loss_meter.update(loss.item(), self.input_config.batch_size)
        self.metric_meters.update(metric_bundle)

    def print_stats(self, epoch, iteration, total_iterations):
        print(
            '{} epoch {}: {}/{}\t'
            'Time {:.3f} ({:.3f})\t'
            'Data {:.3f} ({:.3f})\t'
            'Loss {:.3f} ({:.3f})\t'
            .format(
                self._get_header(), epoch, iteration, total_iterations,
                self.batch_time_meter.val, self.batch_time_meter.avg,
                self.data_time_meter.val, self.data_time_meter.avg,
                self.loss_meter.val, self.loss_meter.avg,
            ),
            end=''
        )
        for metric_label, metric_bundle in self.metric_meters.items():
            print(metric_label, '{:.3f} ({:.3f})\t'.format(
                metric_bundle.val, metric_bundle.avg), end='')
        print()

    def _get_header(self):
        raise NotImplementedError()

    def _parse_data(self, inputs):
        image, label = inputs
        image = Variable(image.cuda(async=True))
        label = Variable(label.cuda(async=True))
        return image, label

    def _get_metrics(self, inputs, targets):
        metric_bundle = meters.MetricBundle({
            metric_label: metric(inputs, targets)
            for metric_label, metric in self.metrics.items()
        })
        return metric_bundle

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        metric_bundle = self._get_metrics(outputs, targets)
        return loss, metric_bundle

    def _step(self, loss, kwargs_):

        raise NotImplementedError()


class Trainer(Runner):
    def _set_model_mode(self):
        self.model.train()

    def _get_header(self):
        return 'Training'

    def _step(self, loss, kwargs_):
        optimizer = kwargs_['optimizer']
        scheduler = kwargs_['scheduler']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


class Evaluator(Runner):
    def _set_model_mode(self):
        self.model.eval()

    def _get_header(self):
        return 'Validation'

    def _step(self, loss, kwargs_):
        return
