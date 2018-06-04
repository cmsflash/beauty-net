import os.path as osp
import argparse
from argparse import Namespace
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from beauty.networks import (
    NetworkFactory, ClassifierFactory, FeatureExtractorFactory
)
from beauty.losses import LossFactory, MetricFactory
from beauty.optimizers import OptimizerFactory
from beauty.lr_schedulers import LrSchedulerFactory
from beauty.datasets import DatasetFactory
from beauty.model_runners import Trainer, Evaluator
from beauty.utils.logging import Logger
from beauty.utils.serialization import save_checkpoint, load_checkpoint


def main(args):
    sys.stdout = Logger(osp.join(args.log_dir, 'log.txt'))
    train_loader = get_data_loader(
        args.dataset,
        args.data_dir,
        args.train_list,
        (args.input_height, args.input_width),
        args.train_resize_method,
        args.batch_size,
        pin_memory=True,
        split='train'
    )
    val_loader = get_data_loader(
        args.dataset,
        args.data_dir,
        args.val_list,
        (args.input_height, args.input_width),
        args.val_resize_method,
        args.batch_size,
        pin_memory=False,
        split='val'
    )
    feature_extractor = FeatureExtractorFactory.create_feature_extractor(
        args.feature_extractor
    )
    classifier = ClassifierFactory.create_classifier(
        args.classifier, feature_extractor.get_feature_channels()
    )
    model = NetworkFactory.create_network(
        args.network, feature_extractor, classifier
    )
    model = nn.DataParallel(model).cuda()
    loss = LossFactory.create_loss(args.loss)
    metrics = MetricFactory.create_metric_bundle(args.metrics)
    trainer = Trainer(model, loss, metrics, args)
    evaluator = Evaluator(model, loss, metrics, args)
    optimizer = OptimizerFactory.create_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
        gradient_threshold=args.gradient_threshold,
        betas=(args.beta1, args.beta2)
    )
    if args.resume_from:
        resume(model, optimizer, args)
        metric_meters = evaluator.run(val_loader, 0)
        for metric_label, metric_meter in metric_meters.items():
            print(metric_label + ': {:5.3}'.format(metric_meter.avg))
    scheduler = LrSchedulerFactory.create_lr_scheduler(
        args.lr_scheduler,
        optimizer,
        start_epoch=args.start_epoch,
        lr_step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    if args.evaluate:
        return

    best_metrics = {metric: 0. for metric in args.metrics}
    for epoch in range(args.start_epoch, args.epochs):
        trainer.run(
            train_loader, epoch, optimizer=optimizer, scheduler=scheduler
        )

        if (epoch + 1) % args.validation_interval == 0:
            metric_meters = evaluator.run(val_loader, epoch)
            log_training(model, optimizer, epoch, metric_meters,
                         best_metrics, args.log_dir)


def get_input_list(input_list_path):
    input_list = []
    with open(input_list_path) as input_list_file:
        for line in input_list_file:
            input_list.append(line)
    return input_list


DATA_LOADER_CONFIGS = {
    'train': Namespace(split_name='Training', shuffle=True, drop_last=True),
    'val': Namespace(split_name='Validatoin', shuffle=False, drop_last=False)
}


def get_data_loader(
    dataset_name, data_dir, data_list_path,
    input_size, resize_method, batch_size, pin_memory, split
):
    data_list = get_input_list(data_list_path)
    config = DATA_LOADER_CONFIGS[split]
    print('{} size: {}'.format(config.split_name, len(data_list)))
    dataset = DatasetFactory.create_dataset(
        dataset_name,
        data_dir,
        data_list,
        input_size,
        transform_method=resize_method
    )
    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=config.shuffle,
        num_workers=1,
        pin_memory=pin_memory,
        drop_last=config.drop_last
    )
    return data_loader


def resume(model, optimizer, args):

    checkpoint = load_checkpoint(args.resume_from)

    model.load_state_dict(checkpoint['state_dict'])
    if not args.refresh_training:
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_metrics = {}
    for metric_label in args.metrics:
        if metric_label in checkpoint:
            best_metrics[metric_label] = checkpoint[metric_label]

    print('=> Start epoch: {:3d}'.format(args.start_epoch), end='')
    for metric_label, metric_value in best_metrics.items():
        print('\tBest {}: {:5.3}'.format(metric_label, metric_value), end='')
    print()


def log_training(model, optimizer, epoch, metric_meters, best_metrics, log_dir):

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
    parser = argparse.ArgumentParser()

    # Operational commanded
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--refresh_training', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0)
    # Data parameters
    parser.add_argument('--dataset')
    parser.add_argument('--data_dir')
    parser.add_argument('--train_list')
    parser.add_argument('--val_list')
    # Model parameters
    parser.add_argument('--network')
    parser.add_argument('--feature_extractor')
    parser.add_argument('--classifier')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='Cross Entropy')
    # Input parameters
    parser.add_argument('--input_height', type=int, default=512)
    parser.add_argument('--input_width', type=int, default=1024)
    parser.add_argument(
        '--train_resize_method', type=str, default='Random Crop'
    )
    parser.add_argument('--val_resize_method', type=str, default='Pad')
    parser.add_argument('--normalization_method', type=str, default='Example')
    # Traiing parametrs
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gradient_threshold', type=float, default=None)
    # Learning rate parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='Constant')
    parser.add_argument('--lr_step_size', type=int, default=100)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    # Auxiliary parameters
    parser.add_argument('--validation_interval', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='logs/default/')
    parser.add_argument('--metrics', nargs='*', type=str, default=['Accuracy'])
    parser.add_argument('--seed', type=int, default=1)
    main(parser.parse_args())
