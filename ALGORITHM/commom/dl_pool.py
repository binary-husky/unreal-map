"""
    Author: Fu Qingxu,CASIA
    Description: deep learning sample manager

"""
import torch
import numpy as np

class DeepLearningPool(object):
    def __init__(self, pool_size, batch_size) -> None:
        super().__init__()
        self.x_batch = None
        self.y_batch = None
        self.size = pool_size
        self.batch_size = batch_size



    def add_and_sample(self, x, y):
        n_sample = x.shape[0]
        assert n_sample > 0
        if self.x_batch is None:
            self.x_batch = np.zeros(shape=(self.size, *x.shape[1:]), dtype=x.dtype)
            self.y_batch = np.zeros(shape=(self.size, *y.shape[1:]), dtype=y.dtype)
            self.current_idx = 0
            self.current_size = 0
        idx = self._get_storage_idx(n_sample)
        self.x_batch[idx] = x
        self.y_batch[idx] = y
        return self._sample()


    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def _sample(self):
        idx = np.random.randint(0, self.current_size, self.batch_size)
        return self.x_batch[idx], self.y_batch[idx]


if __name__ == '__main__':
    dlp = DeepLearningPool(10, 7)
    res = dlp.add_and_sample(x=np.random.rand(2,2,3),y=np.array([1,2]))
    print(dlp.y_batch,'res',res[1])
    res = dlp.add_and_sample(x=np.random.rand(4,2,3),y=np.array([3,4,5,6]))
    print(dlp.y_batch,'res',res[1])
    res = dlp.add_and_sample(x=np.random.rand(3,2,3),y=np.array([7,8,9]))
    print(dlp.y_batch,'res',res[1])
    res = dlp.add_and_sample(x=np.random.rand(3,2,3),y=np.array([10,11,12]))
    print(dlp.y_batch,'res',res[1])
    res = dlp.add_and_sample(x=np.random.rand(3,2,3),y=np.array([13,14,15]))
    print(dlp.y_batch,'res',res[1])
    res = dlp.add_and_sample(x=np.random.rand(3,2,3),y=np.array([16,17,18]))
    print(dlp.y_batch,'res',res[1])
    print('end of test')


        
