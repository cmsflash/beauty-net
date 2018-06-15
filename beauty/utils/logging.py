import os
import os.path as osp
import sys
from io import IOBase

from . import osutils


class Logger(IOBase):
    def __init__(self, dir=None):
        super().__init__()
        self.console = sys.stdout
        self.file = self._get_file(dir)

    def _get_file(self, dir):
        if dir is None:
            path = os.devnull
        else:
            osutils.remove_if_exists(dir)
            osutils.mkdir_if_missing(dir)
            path = osp.join(dir, 'log.txt')
        file = open(path, 'w')
        return file

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        self.file.close()
