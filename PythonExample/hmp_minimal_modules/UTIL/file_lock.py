# pip install filelock
from filelock import FileLock as FileLockBase

class FileLock(FileLockBase):
    def __init__(self, lock_file, timeout: float = -1) -> None:
        assert lock_file.endswith('.lock')
        super().__init__(lock_file, timeout)


def is_file_empty(file_path):
    with open(file_path, 'r') as f: 
        file_content = f.read()
    if file_content == '' or file_content == '\n': 
        return True
    else:
        return False
    