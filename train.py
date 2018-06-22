from os import path as osp
from argparse import Namespace
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from beauty.networks.beauty_net import BeautyNet
from beauty.networks.feature_extractors import *
from beauty.networks.classifiers import *
from beauty import networks
from beauty.losses import MetricFactory
from beauty import lr_schedulers
from beauty import data_loaders
from beauty.datasets import *
from beauty.model_runners import Trainer, Evaluator
from beauty.utils import logging, tensor_utils
from beauty.utils.serialization import save_checkpoint, load_checkpoint


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        config = self.config
        commands = config.commands
        sys.stdout = logging.Logger(config.log_dir)
        device = tensor_utils.get_device()

        train_loader = data_loaders.create_data_loader(
            config.input.train, 'train'
        )
        val_loader = data_loaders.create_data_loader(
            config.input.val, 'val', pin_memory=False
        )
        model = networks.create_model(config.model, device)
        loss = config.model.loss()
        metrics = MetricFactory.create_metric_bundle(config.metrics)
        optimizer = config.optimizer.optimizer(
            model.parameters(), **vars(config.optimizer.config)
        )

        trainer = Trainer(model, loss, metrics, config.input.train)
        evaluator = Evaluator(model, loss, metrics, config.input.val)
        if commands.resume_from:
            self._resume(model, optimizer, config, commands)
        scheduler = lr_schedulers.create_lr_scheduler(config.lr, optimizer)

        if commands.evaluate:
            return

        best_metrics = {metric: 0. for metric in config.metrics}
        for epoch in range(commands.start_epoch, config.training.epochs):
            trainer.run(
                train_loader, epoch, optimizer=optimizer, scheduler=scheduler
            )
            metric_meters = evaluator.run(val_loader, epoch)
            log_training(
                model, optimizer, epoch, metric_meters,
                best_metrics, config.log_dir
            )

    def _resume(self, model, optimizer, config, commands):
        checkpoint = load_checkpoint(commands.resume_from)

        model.load_state_dict(checkpoint['state_dict'])
        if not commands.refresh_training:
            commands.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

        best_metrics = {}
        for metric_label in config.metrics:
            if metric_label in checkpoint:
                best_metrics[metric_label] = checkpoint[metric_label]

        print('=> Start epoch: {:3d}'.format(commands.start_epoch), end='')
        for metric_label, metric_value in best_metrics.items():
            print('\tBest {}: {:5.3}'.format(metric_label, metric_value), end='')
        print()

        metric_meters = evaluator.run(val_loader, 0)
        for metric_label, metric_meter in metric_meters.items():
            print(metric_label + ': {:5.3}'.format(metric_meter.avg))

    def log_training(
            self, model, optimizer, epoch, metric_meters, best_metrics, log_dir
        ):

        are_best = {}
        print('\n * Finished epoch {:3d}:\t'.format(epoch), end='')

        for metric_label, metric_meter in metric_meters.items():
            metric_value = metric_meter.avg

            if metric_value > best_metrics[metric_label]:
                best_metrics[metric_label] = metric_value
                are_best[metric_label] = True
            else:
                are_best[metric_label] = False

            print('{}: {:5.3}\tbest: {:5.3}{}\t'.format(metric_label, metric_value,
                                                        best_metrics[metric_label], ' *' if are_best[metric_label] else ''), end='')

        print()
        print()

        checkpoint = {**{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
            **best_metrics
        }
        save_checkpoint(checkpoint, are_best, log_dir=log_dir)


if __name__ == '__main__':
    gpus = int(sys.argv[1])
    job_name = sys.argv[2]

    config = Namespace(
        commands=Namespace(
            evaluate=False,
            resume_from=None,
            refresh_training=False,
            start_epoch=0
        ),
        input=Namespace(
            train=Namespace(
                dataset=Scut5500Dataset,
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
                dataset=Scut5500Dataset,
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
            feature_extractor=MobileNetV2,
            classifier=SoftmaxClassifier,
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

    ModelTrainer(config).train()
