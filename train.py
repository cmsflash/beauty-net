from os import path as osp
from argparse import Namespace
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from beauty import networks, lr_schedulers, data_loaders, datasets
from beauty.networks.beauty_net import BeautyNet
from beauty.networks import feature_extractors, classifiers
from beauty.losses import MetricFactory
from beauty.model_runners import Trainer, Evaluator
from beauty.utils import tensor_utils, serialization


class ModelTrainer:
    def __init__(self, config, resume_from=None):
        commands = config.commands
        self.config = config
        self.device = tensor_utils.get_device()

        self.train_loader = data_loaders.create_data_loader(
            config.input.train, 'train'
        )
        self.val_loader = data_loaders.create_data_loader(
            config.input.val, 'val', pin_memory=False
        )
        self.model = networks.create_model(config.model, self.device)
        self.loss = config.model.loss()
        self.metrics = MetricFactory.create_metric_bundle(config.metrics)
        self.optimizer = config.optimizer.optimizer(
            self.model.parameters(), **vars(config.optimizer.config)
        )
        self.best_metrics = {metric: 0. for metric in self.config.metrics}

        self.trainer = Trainer(
            self.model, self.loss, self.metrics, config.input.train
        )
        self.evaluator = Evaluator(
            self.model, self.loss, self.metrics, config.input.val
        )
        if resume_from:
            self.resume(self.model, self.optimizer, config, commands)
        self.scheduler = lr_schedulers.create_lr_scheduler(
            config.lr, self.optimizer
        )

    def train(self):
        commands = self.config.commands

        for epoch in range(commands.start_epoch, self.config.training.epochs):
            self.trainer.run(
                self.train_loader, epoch,
                optimizer=self.optimizer, scheduler=self.scheduler
            )
            metric_meters = self.evaluator.run(self.val_loader, epoch)
            log_training(
                self.model, self.optimizer, epoch, metric_meters,
                self.config.log_dir
            )

    def resume(self, commands):
        checkpoint = serialization.load_checkpoint(commands.resume_from)

        self.model.load_state_dict(checkpoint['state_dict'])
        if not commands.refresh_training:
            commands.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        best_metrics_stored = {}
        for metric_label in self.config.metrics:
            if metric_label in checkpoint:
                best_metrics_stored[metric_label] = checkpoint[metric_label]

        print('=> Start epoch: {:3d}'.format(commands.start_epoch), end='')
        for metric_label, metric_value in best_metrics_stored.items():
            print('\tBest {}: {:5.3}'.format(metric_label, metric_value), end='')
        print()

        metric_meters = self.evaluator.run(self.val_loader, 0)
        for metric_label, metric_meter in metric_meters.items():
            print(metric_label + ': {:5.3}'.format(metric_meter.avg))

    def log_training(self, epoch, metric_meters, log_dir):
        are_best = {}
        print('\n * Finished epoch {:3d}:\t'.format(epoch), end='')

        for metric_label, metric_meter in metric_meters.items():
            metric_value = metric_meter.avg

            if metric_value > self.best_metrics[metric_label]:
                self.best_metrics[metric_label] = metric_value
                are_best[metric_label] = True
            else:
                are_best[metric_label] = False

            print(
                '{}: {:5.3}\tbest: {:5.3}{}\t'.format(
                    metric_label, metric_value,
                    self.best_metrics[metric_label],
                    ' *' if are_best[metric_label] else ''
                ), end=''
            )

        print()
        print()

        checkpoint = {
            **{
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            **best_metrics
        }
        serialization.save_checkpoint(checkpoint, are_best, log_dir=log_dir)


if __name__ == '__main__':
    gpus = int(sys.argv[1])
    job_name = sys.argv[2]

    config = Namespace(
        commands=Namespace(
            resume_from=None,
            refresh_training=False,
            start_epoch=0
        ),
        input=Namespace(
            train=Namespace(
                dataset=datasets.Scut5500Dataset,
                config=Namespace(
                    data_dir=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                        'Images/'
                    ),
                    data_list_path=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                        'train_test_files/All_labels.txt'
                    ),
                    input_size=(320, 320),
                    transform_method='Data Augment'
                ),
                batch_size=gpus
            ),
            val=Namespace(
                dataset=datasets.Scut5500Dataset,
                config=Namespace(
                    data_dir=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                        'Images/'
                    ),
                    data_list_path=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                        'train_test_files/All_labels.txt'
                    ),
                    input_size=(320, 320),
                    transform_method='Resize'
                ),
                batch_size=gpus
            )
        ),
        model=Namespace(
            network=BeautyNet,
            feature_extractor=feature_extractors.MobileNetV2,
            classifier=classifiers.SoftmaxClassifier,
            weight_decay=5e-4,
            loss=nn.CrossEntropyLoss
        ),
        training=Namespace(
            epochs=200
        ),
        optimizer=Namespace(
            optimizer=optim.Adam,
            config=Namespace(
                betas=(0.9, 0.99)
            )
        ),
        lr=Namespace(
            lr=1e-5,
            lr_scheduler=lr_schedulers.ConstantLr,
            config=Namespace()
        ),
        log_dir=osp.join('logs', job_name),
        metrics=['Accuracy']
    )

    model_trainer = ModelTrainer(config)
    model_trainer.train()

