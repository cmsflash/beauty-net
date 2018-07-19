import time

from .utils import meters


class ModelMeters:
    def __init__(self, metrics):
        self.batch_time_meter = meters.AverageMeter('Time')
        self.data_time_meter = meters.AverageMeter('Data')
        self.loss_meter = meters.AverageMeter('Loss')
        self.metric_meters = metrics.create_average_meters()

    def reset(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.metric_meters.reset()
    
    def update(
            self, metric_bundle, batch_time=None, data_time=None, loss=None,
            batch_size=1
        ):
        self.batch_time_meter.update(batch_time)
        self.data_time_meter.update(data_time)
        self.loss_meter.update(loss.item(), batch_size)
        self.metric_meters.update(metric_bundle)

    def __str__(self):
        string = (
            f'{self.batch_time_meter}\t{self.data_time_meter}'
            f'\t{self.loss_meter}\t{self.metric_meters}'
        )
        return string


class Runner:
    tag = {True: 'Training', False: 'Validation'}
    training = True

    def __init__(
            self, job_name, model, loss, metrics, device,
            optimizer=None, scheduler=None, input_config=None
        ):
        super().__init__()
        self.job_name = job_name
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.input_config = input_config

        self.meters = ModelMeters(metrics)

    def run(self, data_loader, epoch):
        self._set_model_mode()
        self._epoch_step()
        self.meters.reset()
        start_time = time.time()
        for i, inputs in enumerate(data_loader):
            self._iterate(i, inputs, epoch, len(data_loader), start_time)
            start_time = time.time()
        return self.meters.metric_meters

    def _iterate(self, i, inputs, epoch, loader_length, start_time):
        data_time = time.time() - start_time
        inputs, targets = self._parse_data(inputs)
        loss, metric_bundle = self._forward(inputs, targets)
        self._step(loss)
        batch_time = time.time() - start_time
        self.meters.update(
            metric_bundle, batch_time, data_time, loss,
            self.input_config.batch_size
        )
        self.print_stats(epoch, i + 1, loader_length)
        start_time = time.time()

    def _set_model_mode(self):
        self.model.train(self.training)

    def _epoch_step(self):
        pass

    def print_stats(self, epoch, iteration, total_iterations):
        print(
            f'{self._get_header(epoch, iteration, total_iterations)}'
            f'\t{self.meters}'
        )

    def _get_header(self, epoch, iteration, total_iterations):
        header = '{} epoch {}: {}/{}'.format(
            self.tag[self.training], epoch, iteration, total_iterations
        )
        return header

    def _parse_data(self, inputs):
        image, label = inputs
        image = image.to(self.device)
        label = label.to(self.device)
        return image, label

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        metric_bundle = self.metrics(outputs, targets)
        return loss, metric_bundle

    def _step(self, loss):
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()


class Trainer(Runner):
    pass


class Evaluator(Runner):
    training = False
