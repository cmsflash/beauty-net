import os.path as osp
import shutil

import torch

from . import osutils


def save_checkpoint(checkpoint, log_dir):
    osutils.mkdir_if_missing(log_dir)
    checkpoint_file = osp.join(log_dir, 'checkpoint.pth.tar')
    torch.save(checkpoint, checkpoint_file)
    are_best = {
        label: meter.latest
        for label, meter in checkpoint['best_meters'].meters.items()
    }
    for metric_name, is_best in are_best.items():
        if is_best:
            shutil.copy(checkpoint_file, osp.join(
                log_dir, 'best_' + metric_name + '.pth.tar'
            ))
