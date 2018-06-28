from os import path as osp
from argparse import Namespace
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from beauty import networks, metrics, lr_schedulers, data_loaders, datasets
from beauty.networks.beauty_net import BeautyNet
from beauty.networks import feature_extractors, classifiers
from beauty.model_runners import Trainer, Evaluator
from beauty.utils import tensor_utils, serialization, meters


class ModelTrainer:
    def __init__(self, config, resume_from=None):
        self.config = config
        self.start_epoch = 0
        self.device = tensor_utils.get_device()

        self.train_loader = data_loaders.create_data_loader(
            config.input.train, 'train'
        )
        self.val_loader = data_loaders.create_data_loader(
            config.input.val, 'val', pin_memory=False
        )
        self.model = networks.create_model(config.model, self.device)
        self.loss = config.model.loss()
        self.metrics = metrics.create_metric_bundle(config.metrics)
        self.optimizer = config.optimizer.optimizer(
            self.model.parameters(), **vars(config.optimizer.config)
        )
        self.best_meters = meters.MeterBundle(
            meters.MaxMeter(metric) for metric in self.config.metrics
        )

        self.trainer = Trainer(
            self.model, self.loss, self.metrics, config.input.train
        )
        self.evaluator = Evaluator(
            self.model, self.loss, self.metrics, config.input.val
        )
        self.scheduler = lr_schedulers.create_lr_scheduler(
            config.lr, self.optimizer
        )

    def train(self):
        for epoch in range(self.start_epoch, self.config.training.epochs):
            self.trainer.run(
                self.train_loader, epoch,
                optimizer=self.optimizer, scheduler=self.scheduler
            )
            metric_meters = self.evaluator.run(self.val_loader, epoch)
            self.log_training(epoch, metric_meters, self.config.log_dir)

    def resume(self, checkpoint_path, refresh=True):
        checkpoint = serialization.load_checkpoint(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if not refresh:
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Training resumed')
        print('Start epoch: {:3d}'.format(self.start_epoch))
        print('Best metrics: {}'.format(checkpoint['best_meters']))

    def log_training(self, epoch, metric_meters, log_dir):
        are_best = {}
        print('\n * Finished epoch {:3d}:\t'.format(epoch), end='')

        self.best_meters.update(metric_meters)
        for metric_label, metric_meter in metric_meters.items():
            print(best_meter, end='')

        print()
        print()

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_meters': self.best_meters
        }
        serialization.save_checkpoint(checkpoint, are_best, log_dir=log_dir)


if __name__ == '__main__':
    gpus = int(sys.argv[1])
    job_name = sys.argv[2]

    config = Namespace(
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
                        'train_test_files/1_label.txt'
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
                        'train_test_files/1_label.txt'
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
        metrics=[metrics.Accuracy]
    )

    model_trainer = ModelTrainer(config)
    model_trainer.train()
