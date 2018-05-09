import os.path as osp
import shutil

import torch

from .osutils import mkdir_if_missing


def save_checkpoint(state, are_best, log_dir='log/default'):
    mkdir_if_missing(log_dir)
    checkpoint_file = osp.join(log_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_file)
    for metric_name, is_best in are_best.items():
        if is_best:
            shutil.copy(checkpoint_file, osp.join(
                log_dir, 'best_' + metric_name + '.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
