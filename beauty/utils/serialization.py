from os import path as osp
import shutil

import torch

from . import os_utils


def save_checkpoint(checkpoint, log_config):
    os_utils.make_dir_if_missing(log_config.dir)
    checkpoint_path = osp.join(log_config.dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    are_best = {
        label: meter.latest
        for label, meter in checkpoint['best_meters'].meters.items()
    }
    for metric_name, is_best in are_best.items():
        if is_best:
            shutil.copy(checkpoint_path, osp.join(
                log_config.dir, f'best_{metric_name}.pth'
            ))
    epoch = checkpoint['epoch']
    if epoch % log_config.interval == 0:
        shutil.copy(
            checkpoint_path, osp.join(log_config.dir, f'epoch{epoch - 1}.pth')
        )
