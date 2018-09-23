import os
from os import path as osp
import shutil


def make_dir_if_missing(dir_):
    if not osp.exists(dir_):
        os.makedirs(dir_)


def remove_if_exists(path):
    if osp.exists(path):
        shutil.rmtree(path)
