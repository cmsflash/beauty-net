import os
import errno
import shutil


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def remove_if_exists(path):
    try:
        shutil.rmtree(path)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
