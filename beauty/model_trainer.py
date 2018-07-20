from . import networks, metrics, lr_schedulers, data_loaders
from .model_runners import Runner
from .utils import tensor_utils, serialization


class ModelTrainer:
    def __init__(self, job_name, config, resume_from=None):
        self.job_name = job_name
        self.config = config
        self.start_epoch = 0
        self.epoch = self.start_epoch
        self.device = tensor_utils.get_device()

        self.train_loader = data_loaders.create_data_loader(
            config.input.train, data_loaders.TRAIN_CONFIG
        )
        self.val_loader = data_loaders.create_data_loader(
            config.input.val, data_loaders.VAL_CONFIG, pin_memory=False
        )
        self.model = networks.create_model(config.model, self.device)
        self.loss = config.model.loss()
        self.metrics = metrics.create_metric_bundle(config.metrics)
        self.optimizer = config.optimizer.optimizer(
            self.model.parameters(), **vars(config.optimizer.config)
        )
        self.scheduler = lr_schedulers.create_lr_scheduler(
            config.lr, self.optimizer
        )
        self.best_meters = self.metrics.create_max_meters()

        self.trainer = Runner(
            self.job_name, self.model, self.loss, self.metrics, self.device,
            self.optimizer, self.scheduler
        )

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.epochs):
            self.epoch = epoch
            self.trainer.train(True)
            self.trainer.run(self.train_loader, self.epoch)
            self.trainer.train(False)
            metric_meters = self.trainer.run(self.val_loader, self.epoch)
            self.log_training(metric_meters, self.config.log_dir)

    def resume(self, checkpoint_path, refresh=True):
        checkpoint = serialization.load_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if not refresh:
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Training resumed')
        print('Start epoch: {:3d}'.format(self.start_epoch))
        print('Best metrics: {}'.format(checkpoint['best_meters']))

    def log_training(self, metric_meters, log_dir):
        print('\n * Finished epoch {:3d}:\t'.format(self.epoch), end='')

        self.best_meters.update(metric_meters)
        are_best = {
            label: meter.latest
            for label, meter in self.best_meters.meters.items()
        }
        print(self.best_meters)
        print()
        print()

        checkpoint = {
            'epoch': self.epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_meters': self.best_meters
        }
        serialization.save_checkpoint(checkpoint, are_best, log_dir=log_dir)
