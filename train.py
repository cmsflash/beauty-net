import os.path as osp
from argparse import Namespace
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from beauty.networks.beauty_net import BeautyNet
from beauty.networks.feature_extractors import *
from beauty.networks.classifiers import *
from beauty.losses import MetricFactory
from beauty.datasets import *
from beauty.model_runners import Trainer, Evaluator
from beauty.utils.logging import Logger
from beauty.utils.serialization import save_checkpoint, load_checkpoint


class ModelTrainer:
    CLASS_COUNT = 5
    DATA_LOADER_CONFIGS = {
        'train': Namespace(split_name='Training', shuffle=True, drop_last=True),
        'val': Namespace(
            split_name='Validatoin', shuffle=False, drop_last=False
        )
    }

    def __init__(self, args):
        self.args = args

    def train(self):
        args = self.args
        commands = args.commands
        sys.stdout = self._get_logger(args.log_dir)
        device = self._get_device()

        train_loader = self._get_data_loader(
            args.data.train, args.input.train, split='train'
        )
        val_loader = self._get_data_loader(
            args.data.val, args.input.val, split='val', pin_memory=False
        )
        model = self._get_model(args.model, device)
        loss = self._get_loss(args.model)
        metrics = self._get_metrics(args.metrics)
        optimizer = self._get_optimizer(args.optimizer, model.parameters())

        trainer = Trainer(model, loss, metrics, args.input.train)
        evaluator = Evaluator(model, loss, metrics, args.input.val)
        if commands.resume_from:
            self._resume(model, optimizer, args, commands)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)

        if commands.evaluate:
            return

        best_metrics = {metric: 0. for metric in args.metrics}
        for epoch in range(commands.start_epoch, args.training.epochs):
            trainer.run(
                train_loader, epoch, optimizer=optimizer, scheduler=scheduler
            )
            metric_meters = evaluator.run(val_loader, epoch)
            log_training(model, optimizer, epoch, metric_meters,
                         best_metrics, args.log_dir)

    def _get_logger(self, log_dir):
        logger = Logger(log_dir)
        return logger

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def _get_input_list(self, input_list_path):
        input_list = []
        with open(input_list_path) as input_list_file:
            for line in input_list_file:
                input_list.append(line)
        return input_list

    def _get_dataset(self, data_config, input_config):
        data_list = self._get_input_list(data_config.data_list_path)
        dataset = data_config.dataset(
            data_config.data_dir,
            data_list,
            input_config.input_size,
            transform_method=input_config.resize_method
        )
        return dataset

    def _get_data_loader(
        self, data_config, input_config, split, pin_memory=True
    ):
        loader_config = self.DATA_LOADER_CONFIGS[split]
        dataset = self._get_dataset(data_config, input_config)
        print('{} size: {}'.format(loader_config.split_name, len(dataset)))
        data_loader = DataLoader(
            dataset,
            input_config.batch_size,
            shuffle=loader_config.shuffle,
            num_workers=1,
            pin_memory=pin_memory,
            drop_last=loader_config.drop_last
        )
        return data_loader

    def _get_model(self, model_config, device):
        feature_extractor = model_config.feature_extractor()
        classifier = model_config.classifier(
            feature_extractor.get_feature_channels(), self.CLASS_COUNT
        )
        model = model_config.network(feature_extractor, classifier)
        model = nn.DataParallel(model).to(device)
        return model

    def _get_loss(self, model_config):
        loss = model_config.loss()
        return loss

    def _get_metrics(self, metrics):
        metrics = MetricFactory.create_metric_bundle(metrics)
        return metrics

    def _get_optimizer(self, optimizer_config, parameters):
        optimizer = optimizer_config.optimizer(
            parameters, **vars(optimizer_config.parameters)
        )
        return optimizer

    def _resume(self, model, optimizer, args, commands):
        checkpoint = load_checkpoint(commands.resume_from)

        model.load_state_dict(checkpoint['state_dict'])
        if not commands.refresh_training:
            commands.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

        best_metrics = {}
        for metric_label in args.metrics:
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
        data=Namespace(
            train=Namespace(
                dataset=Scut5500Dataset,
                data_dir=(
                    '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                    'Images/'
                ),
                data_list_path=(
                    '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                    'train_test_files/All_labels.txt'
                ),
            ),
            val=Namespace(
                dataset=Scut5500Dataset,
                data_dir=(
                    '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                    'Images/'
                ),
                data_list_path=(
                    '/mnt/lustre/share/shenzhuoran/datasets/scut-fbp5500/'
                    'train_test_files/All_labels.txt'
                )
            )
        ),
        input=Namespace(
            train=Namespace(
                input_size=(320, 320),
                resize_method='Data Augment',
                batch_size=gpus,
            ),
            val=Namespace(
                input_size=(320, 320),
                resize_method='Resize',
                batch_size=gpus,
            ),
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
            parameters=Namespace(
                betas=(0.9, 0.99)
            )
        ),
        lr=Namespace(
            lr=1e-5,
            lr_scheduler=optim.lr_scheduler.StepLR,
            lr_step_size=100,
            lr_gamma=0.1
        ),
        log_dir=osp.join('logs', job_name),
        metrics=['Accuracy']
    )

    ModelTrainer(config).train()
