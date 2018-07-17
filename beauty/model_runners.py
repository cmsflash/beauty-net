import time

from torch.autograd import Variable

from .utils import meters


class Runner:
    tag = None
    training = True

    def __init__(self, job_name, model, loss, metrics, input_config):
        super().__init__()
        self.job_name = job_name
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.input_config = input_config

        self.batch_time_meter = meters.AverageMeter()
        self.data_time_meter = meters.AverageMeter()
        self.loss_meter = meters.AverageMeter()
        self.metric_meters = metrics.create_average_meters()

    def run(self, data_loader, epoch, optimizer=None, scheduler=None):
        self._set_model_mode()
        self._reset_stats()

        start_time = time.time()
        for i, inputs in enumerate(data_loader):
            self._iterate(
                i, inputs, epoch, len(data_loader), start_time,
                optimizer, scheduler
            )
        return self.metric_meters

    def _iterate(
            self, i, inputs, epoch, loader_length, start_time,
            optimizer, scheduler
        ):
        data_time = time.time() - start_time
        inputs, targets = self._parse_data(inputs)
        loss, metric_bundle = self._forward(inputs, targets)
        self._step(loss, optimizer, scheduler)
        batch_time = time.time() - start_time
        self._update_stats(batch_time, data_time, loss, metric_bundle)
        self.print_stats(epoch, i + 1, loader_length)
        start_time = time.time()

    def _set_model_mode(self):
        self.model.train(self.training)

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
            '{}\t{}\t{}\t'
            .format(
                self.tag, epoch, iteration, total_iterations,
                self.batch_time_meter, self.data_time_meter, self.loss_meter
            ),
            end=''
        )
        print(self.metric_meters)

    def _parse_data(self, inputs):
        image, label = inputs
        image = Variable(image.cuda(async=True))
        label = Variable(label.cuda(async=True))
        return image, label

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        metric_bundle = self.metrics(outputs, targets)
        return loss, metric_bundle

    def _step(self, loss, optimizer, scheduler):
        return


class Trainer(Runner):
    tag = 'Training'

    def _step(self, loss, optimizer, scheduler):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


class Evaluator(Runner):
    tag = 'Validation'
    training = False
