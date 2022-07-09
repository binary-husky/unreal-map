import platform, os, torch, uuid, time
from atexit import register
IsLinux = platform.system()=="Linux"
if IsLinux: 
    print('system is not Linux, flock module not available!')
    import flock # flock is Linux only

class GpuShareUnit():
    def __init__(self, which_gpu, lock_path=None, manual_gpu_ctl=True, gpu_party=''):
        self.device = which_gpu
        self.manual_gpu_ctl = True
        self.lock_path=lock_path
        self.gpu_party = gpu_party
        self.experiment_uuid = uuid.uuid1().hex + '\n'
        self.n_gpu_process_online = 1
        if gpu_party == 'off':
            self.manual_gpu_ctl = False
        if self.lock_path is None: 
            self.lock_path = os.path.expanduser('~/GpuLock')
        if not os.path.exists(self.lock_path): os.makedirs(self.lock_path)
        register(self.__del__)
        
    def __del__(self):
        if hasattr(self,'_deleted_'): return    # avoid exit twice
        else: self._deleted_ = True     # avoid exit twice

        self.unregister_uuid_()
        try: self.gpu_lock.__exit__(None,None,None)
        except:pass
        try: self.gpu_lock_file.close()
        except:pass

    def __enter__(self):
        self.get_gpu_lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_gpu_lock()

    def get_gpu_lock(self):
        if self.manual_gpu_ctl:
            print('Waiting for GPU %s %s...'%(self.device, self.gpu_party), end='', flush=True)
            self.gpu_lock, self.gpu_lock_file = (None, None)
            self.n_gpu_process_online = self.register_uuid_()
            self.gpu_lock_file = open(self.lock_path+'/lock_gpu_%s_%s.glock'%(self.device, self.gpu_party), 'w+')
            self.gpu_lock = flock.Flock(self.gpu_lock_file, flock.LOCK_EX)
            self.gpu_lock.__enter__()
            print('Get GPU, currently shared with %d process!'%self.n_gpu_process_online)
        return

    def release_gpu_lock(self):
        if self.manual_gpu_ctl:
            if self.n_gpu_process_online >1: 
                torch.cuda.empty_cache()
                self.gpu_lock.__exit__(None,None,None)
                self.gpu_lock_file.close()
            else:
                print('不共享GPU')
        return
    
    def register_uuid_(self):
        try:
            flag = 'w+' if not os.path.exists(self.lock_path+'/lock_gpu_%s_%s.register'%(self.device, self.gpu_party)) else 'r+'
            with open(self.lock_path+'/lock_gpu_%s_%s.register'%(self.device, self.gpu_party), mode=flag) as gpu_register_file:
                _lock = flock.Flock(gpu_register_file, flock.LOCK_EX); _lock.__enter__()
                lines = gpu_register_file.readlines()
                if not any([line==self.experiment_uuid for line in lines]):
                    lines.append(self.experiment_uuid)
                    gpu_register_file.seek(0); gpu_register_file.truncate(0)
                    gpu_register_file.writelines(lines)
                    gpu_register_file.flush()
                _lock.__exit__(None,None,None)
                return len(lines)
        except:
            print('GPU 队列异常!')
            return 999

    def unregister_uuid_(self):
        flag = 'w+' if not os.path.exists(self.lock_path+'/lock_gpu_%s_%s.register'%(self.device, self.gpu_party)) else 'r+'
        with open(self.lock_path+'/lock_gpu_%s_%s.register'%(self.device, self.gpu_party), mode=flag) as gpu_register_file:
            _lock = flock.Flock(gpu_register_file, flock.LOCK_EX); _lock.__enter__()
            lines = gpu_register_file.readlines()
            gpu_register_file.seek(0); gpu_register_file.truncate(0)
            gpu_register_file.writelines([line for line in lines if line!=self.experiment_uuid])
            gpu_register_file.flush()
            _lock.__exit__(None,None,None)
            print('unregister')

if not IsLinux: 
    class GpuShareUnitFake():
        def __init__(self, which_gpu, lock_path=None, manual_gpu_ctl=True, gpu_party=''):
            print('system is not Linux, Using a fake GpuShareUnit!')

        def __del__(self):
            return

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return
    GpuShareUnit = GpuShareUnitFake