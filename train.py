from os import path as osp
from argparse import Namespace
import sys

from torch import nn, optim
from torchvision import transforms

from beauty import Task, networks, metrics, lr_schedulers, datasets


if __name__ == '__main__':
    gpus = int(sys.argv[1])
    task_name = sys.argv[2]

    config = Namespace(
        data=Namespace(
            train=Namespace(
                dataset=datasets.Scut5500Dataset,
                config=Namespace(
                    data_dir=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut_fbp5500/'
                        'Images/'
                    ),
                    data_list_path=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut_fbp5500/'
                        'train_test_files/All_labels.txt'
                    ),
                    transforms=[
                        Namespace(
                            transform=transforms.Resize,
                            config=Namespace(size=(320, 320))
                        ),
                        Namespace(
                            transform=transforms.ToTensor,
                            config=Namespace()
                        )
                    ],
                    discrete=True
                ),
                batch_size=gpus * 32
            ),
            val=Namespace(
                dataset=datasets.Scut5500Dataset,
                config=Namespace(
                    data_dir=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut_fbp5500/'
                        'Images/'
                    ),
                    data_list_path=(
                        '/mnt/lustre/share/shenzhuoran/datasets/scut_fbp5500/'
                        'train_test_files/All_labels.txt'
                    ),
                    transforms=[
                        Namespace(
                            transform=transforms.Resize,
                            config=Namespace(size=(320, 320))
                        ),
                        Namespace(
                            transform=transforms.ToTensor,
                            config=Namespace()
                        )
                    ],
                    discrete=True
                ),
                batch_size=gpus * 32
            )
        ),
        model=Namespace(
            network=networks.BeautyNet,
            feature_extractor=networks.feature_extractors.MobileNetV2,
            classifier=networks.classifiers.SoftmaxClassifier,
            class_count=5,
            weight_decay=5e-4,
            loss=nn.CrossEntropyLoss
        ),
        training=Namespace(
            epochs=1000
        ),
        optimizer=Namespace(
            optimizer=optim.Adam,
            config=Namespace(
                betas=(0.9, 0.99)
            )
        ),
        lr=Namespace(
            lr=1e-3,
            lr_scheduler=lr_schedulers.ConstantLr,
            config=Namespace()
        ),
        log=Namespace(
            dir=osp.join('logs', task_name),
            interval=1,
            metrics=[metrics.Accuracy]
        )

    )

    task = Task(task_name, config)
    task.train()
