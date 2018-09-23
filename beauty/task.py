import time

import torch

from . import networks, metrics, data_loaders, utils


class Task:
    tags = {True: 'training', False: 'validation'}

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.epoch = -1
        self.iteration = -1
        self.training = True
        self.device = utils.tensor_utils.get_device()

        self.loaders = {
            True: data_loaders.create_data_loader(
                config.data.train, data_loaders.TRAIN_CONFIG
            ),
            False: data_loaders.create_data_loader(
                config.data.val, data_loaders.VAL_CONFIG, pin_memory=False
            )
        }
        self.model = networks.create_model(config.model, self.device)
        self.loss = config.model.loss()
        self.metrics = metrics.create_metric_bundle(config.log.metrics)
        self.meters = utils.meters.ModelMeters(self.metrics)
        self.optimizer = config.optimizer.optimizer(
            self.model.parameters(), **vars(config.optimizer.config)
        )
        self.scheduler = config.lr.lr_scheduler(
            self.optimizer, **vars(config.lr.config)
        )
        self.best_meters = self.metrics.create_max_meters()

    def train(self):
        start_epoch = self.epoch + 1
        for epoch in range(start_epoch, self.config.training.epochs):
            self.epoch = epoch
            self._run_epoch(training=True)
            metric_meters = self._run_epoch(training=False)
            self._log_training(metric_meters)

    def resume(self, checkpoint_path, refresh=True, partial=False):
        checkpoint = torch.load(checkpoint_path)
        if partial:
            model_state_dict = self.model.state_dict()
            model_state_dict.update(checkpoint['model'])
        else:
            model_state_dict = checkpoint['model']
        self.model.load_state_dict(model_state_dict)
        if not refresh:
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Training resumed at epoch {self.epoch}')
        print(f'Best metrics: {checkpoint["best_meters"]}')

    def set_training(self, training):
        self.training = training
        self.model.train(self.training)

    def _run_epoch(self, training=None):
        if training is not None:
            self.set_training(training)
        self._epoch_step()
        self.meters.reset()
        loader = self.loaders[self.training]
        start_time = time.time()
        for data in loader:
            self._iterate(data, start_time)
            start_time = time.time()
        return self.meters.metric_meters

    def _epoch_step(self):
        self.iteration = -1
        self.scheduler.step()

    def _iterate(self, data, start_time):
        self.iteration += 1
        _, input_, target = self._parse_data(data)
        loss, metric_bundle = self._forward(input_, target)
        self._step(loss)
        iteration_time = time.time() - start_time
        self.meters.update(
            iteration_time, loss, metric_bundle, batch_size=input_.size(0)
        )
        self._print_stats()

    def _parse_data(self, data):
        index, input_, target = data
        input_ = input_.to(self.device)
        target = target.to(self.device)
        return index, input_, target

    def _forward(self, input_, target):
        output = self.model(input_)
        loss = self.loss(output, target)
        metric_bundle = self.metrics(output, target)
        return loss, metric_bundle

    def _step(self, loss):
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _print_stats(self):
        print(f'{self._get_header()}\t{self.meters}')

    def _get_header(self):
        header = (
            f'Epoch {self.epoch} {self.tags[self.training]}'
            f' {self.iteration}/{len(self.loaders[self.training])}:'
        )
        return header

    def _log_training(self, metric_meters):
        self.best_meters.update(metric_meters)
        print(f'\n * Finished epoch {self.epoch}:\t{self.best_meters}\n\n')
        checkpoint = {
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_meters': self.best_meters
        }
        utils.serialization.save(checkpoint, self.config.log)
