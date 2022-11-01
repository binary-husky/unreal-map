# pip install filelock
from filelock import FileLock as FileLockBase

class FileLock(FileLockBase):
    def __init__(self, lock_file, timeout: float = -1) -> None:
        assert lock_file.endswith('.lock')
        super().__init__(lock_file, timeout)