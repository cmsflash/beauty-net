import os
import sys
from io import IOBase

from .osutils import mkdir_if_missing


class NoneIO(IOBase):
    def __init__(self):
        return

    def read(self):
        return

    def write(self, *_, **__):
        return
    

class Logger(object):
    def __init__(self, log_path=None):
        self.console = sys.stdout
        self.file = None
        if log_path is not None:
            mkdir_if_missing(os.path.dirname(log_path))
            self.file = open(log_path, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
