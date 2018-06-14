import os
import os.path as osp
import sys
from io import IOBase

from .osutils import mkdir_if_missing, remove_if_exists


class Logger(IOBase):
    def __init__(self, log_path=os.devnull):
        super().__init__()
        self.console = sys.stdout
        self.file = self._get_file(log_path)

    def _get_file(self, path):
        remove_if_exists(osp.dirname(path))
        mkdir_if_missing(osp.dirname(path))
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
