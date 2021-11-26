import flock, os, torch

class gpu_share_unit():
    def __init__(self, which_gpu, lock_path=None, manual_gpu_ctl=True, gpu_party=''):
        self.device = which_gpu
        self.manual_gpu_ctl = True
        self.lock_path=lock_path
        self.gpu_party = gpu_party
        if gpu_party == 'off':
            self.manual_gpu_ctl = False
        
        if self.lock_path is None: 
            self.lock_path='./RECYCLE/' # os.path.dirname(__file__)
            
        if not os.path.exists(self.lock_path): os.makedirs(self.lock_path)

    def __enter__(self):
        self.get_gpu_lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_gpu_lock()

    def get_gpu_lock(self):
        if self.manual_gpu_ctl:
            print('Waiting for GPU %s %s...'%(self.device, self.gpu_party), end='', flush=True)
            self.gpu_lock, self.gpu_lock_file = (None, None)
            self.gpu_lock_file = open(self.lock_path+'/lock_gpu_%s_%s.glock'%(self.device, self.gpu_party), 'w+')
            self.gpu_lock = flock.Flock(self.gpu_lock_file, flock.LOCK_EX)
            self.gpu_lock.__enter__()
            print('Get GPU!')
        return

    def release_gpu_lock(self):
        if self.manual_gpu_ctl:
            torch.cuda.empty_cache()
            self.gpu_lock.__exit__(None,None,None)
            self.gpu_lock_file.close()
        return