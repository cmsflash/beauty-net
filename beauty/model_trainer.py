import time

from . import networks, metrics, lr_schedulers, data_loaders, utils


class ModelTrainer:
    tags = {True: 'Training', False: 'Validation'}

    def __init__(self, job_name, config, resume_from=None):
        self.job_name = job_name
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
            self.run_epoch(training=True)
            metric_meters = self.run_epoch(training=False)
            self.log_training(metric_meters)

    def resume(self, checkpoint_path, refresh=True):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if not refresh:
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Training resumed at epoch {self.epoch}')
        print(f'Best metrics: {checkpoint["best_meters"]}')

    def log_training(self, metric_meters):
        print(f'\n * Finished epoch {self.epoch}:\t', end='')
        self.best_meters.update(metric_meters)
        are_best = {
            label: meter.latest
            for label, meter in self.best_meters.meters.items()
        }
        print(f'{self.best_meters}\n\n')

        checkpoint = {
            'epoch': self.epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_meters': self.best_meters
        }
        utils.serialization.save_checkpoint(checkpoint, self.config.log)

    def run_epoch(self, training=None):
        if training is not None:
            self.set_training(training)
        self._epoch_step()
        self.meters.reset()
        loader = self.loaders[self.training]
        start_time = time.time()
        for inputs in loader:
            self._iterate(inputs, start_time)
            start_time = time.time()
        return self.meters.metric_meters
        
    def set_training(self, training):
        self.training = training
        self.model.train(self.training)

    def _iterate(self, inputs, start_time):
        self.iteration += 1
        data_time = time.time() - start_time
        inputs, targets = self._parse_data(inputs)
        loss, metric_bundle = self._forward(inputs, targets)
        self._step(loss)
        batch_time = time.time() - start_time
        self.meters.update(
            metric_bundle, batch_time, data_time, loss,
            batch_size=inputs.size(0)
        )
        self.print_stats()
        start_time = time.time()
        
    def _epoch_step(self):
        pass

    def print_stats(self):
        print(f'{self._get_header()}\t{self.meters}')

    def _get_header(self):
        header = (
            f'{self.tags[self.training]} epoch {self.epoch}:'
            f' {self.iteration}/{len(self.loaders[self.training])}'
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
