import flock, os, torch, uuid, time, glob
from atexit import register

class GpuShareUnit():
    def __init__(self, which_gpu, lock_path=None, manual_gpu_ctl=True, gpu_party=''):
        self.device = which_gpu
        self.manual_gpu_ctl = True
        self.lock_path=lock_path
        self.gpu_party = gpu_party
        self.experiment_uuid = uuid.uuid1().hex + '\n'
        self.n_gpu_process_online = 1
        self.flag_req_all_party = False
        self.parties_req = None #   如果单个party的显存不够用，那么需要同时排队多个party，获取内存
        if gpu_party == 'off' or gpu_party == 'OFF' or gpu_party<0:
            self.manual_gpu_ctl = False
        if self.lock_path is None: 
            self.lock_path = os.path.expanduser('~/GpuLock')
        if not os.path.exists(self.lock_path): os.makedirs(self.lock_path)
        register(self.unregister_uuids_)
        

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.flag_req_all_party:
            self.release_gpu_lock()

    def __enter__(self):
        self._get_gpu_locks()
        return self

    def __del__(self):
        self.unregister_uuids_()

    def _get_gpu_locks(self):
        if not self.flag_req_all_party:
            self.parties_req = None
            self.__get_gpu_lock(self.device, self.gpu_party)
        else:
            self.parties_req = self.__find_all_active_party(self.device)
            if not (self.gpu_party in self.parties_req): self.parties_req.append(self.gpu_party)
            for each_party in self.parties_req: self.__get_gpu_lock(self.device, each_party)

    def __find_all_active_party(self, device):
        list_of_active_parties = []
        for indx in range(64):
            res = self.___get_party_n_share(device, gpu_party=str(indx))
            if res is None:
                break
            if res == 0:
                break
            if res >0:
                list_of_active_parties.append(str(indx))
        return list_of_active_parties


    def __get_gpu_lock(self, device, gpu_party):
        if self.manual_gpu_ctl:
            print('Waiting for GPU %s %s...'%(device, gpu_party), end='', flush=True)
            gpu_lock, gpu_lock_file = (None, None)
            self.n_gpu_process_online = self.register_uuid_(device, gpu_party)
            self.gpu_lock_file = open(self.lock_path+'/lock_gpu_%s_%s.glock'%(device, gpu_party), 'w+')
            self.gpu_lock = flock.Flock(self.gpu_lock_file, flock.LOCK_EX)
            self.gpu_lock.__enter__()
            print('Get GPU, currently shared with %d process!'%self.n_gpu_process_online)
        return

    def release_gpu_lock(self):
        self.flag_req_all_party = False
        if self.manual_gpu_ctl:
            if self.n_gpu_process_online >1: 
                torch.cuda.empty_cache()
                self.gpu_lock.__exit__(None,None,None)
                self.gpu_lock_file.close()
            else:
                print('不共享GPU')
        return
    
    def ___get_party_n_share(self, device, gpu_party):
        try:
            flag = 'r'
            with open(self.lock_path+'/lock_gpu_%s_%s.register'%(device, gpu_party), mode=flag) as gpu_register_file:
                _lock = flock.Flock(gpu_register_file, flock.LOCK_EX); _lock.__enter__()
                lines = gpu_register_file.readlines()
                _lock.__exit__(None,None,None)
                return len(lines)
        except: 
            return None

    def register_uuid_(self, device, gpu_party):
        try:
            flag = 'w+' if not os.path.exists(self.lock_path+'/lock_gpu_%s_%s.register'%(device, gpu_party)) else 'r+'
            with open(self.lock_path+'/lock_gpu_%s_%s.register'%(device, gpu_party), mode=flag) as gpu_register_file:
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

    def unregister_uuids_(self, device, gpu_party):
        for __
            self.unregister_uuid_(device, gpu_party)
        try: self.gpu_lock.__exit__(None,None,None)
        except:pass
        try: self.gpu_lock_file.close()
        except:pass


    def unregister_uuid__(self, device, gpu_party):
        flag = 'w+' if not os.path.exists(self.lock_path+'/lock_gpu_%s_%s.register'%(device, gpu_party)) else 'r+'
        with open(self.lock_path+'/lock_gpu_%s_%s.register'%(device, gpu_party), mode=flag) as gpu_register_file:
            _lock = flock.Flock(gpu_register_file, flock.LOCK_EX); _lock.__enter__()
            lines = gpu_register_file.readlines()
            gpu_register_file.seek(0); gpu_register_file.truncate(0)
            gpu_register_file.writelines([line for line in lines if line!=self.experiment_uuid])
            gpu_register_file.flush()
            _lock.__exit__(None,None,None)
            print('unregister')

    def req_all_party(self):
        self.flag_req_all_party = True