from os import path as osp
from argparse import Namespace
import sys

from torch import nn, optim

from beauty import ModelTrainer, networks, metrics, lr_schedulers, datasets
from beauty.networks.beauty_net import BeautyNet


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
            feature_extractor=networks.feature_extractors.MobileNetV2,
            classifier=networks.classifiers.SoftmaxClassifier,
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
