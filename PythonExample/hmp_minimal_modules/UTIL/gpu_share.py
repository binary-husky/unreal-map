import platform, os, torch, uuid, time, psutil, json, random
from UTIL.network import UnixTcpClientP2P, UnixTcpServerP2P
from atexit import register
from .file_lock import FileLock
def pid_exist(pid_str):
    pid = int(pid_str)
    return psutil.pid_exists(pid)

def read_json(fp):
    # create if not exist
    if not os.path.exists(fp):
        with open(fp, "w") as f:
            pass
    # try to read, otherwise reset
    try:
        with open(fp, "r+") as f: 
            json_data = json.load(f)
    except:
        json_data = {}
    return json_data

def write_json(fp, buf):
    with open(fp, "w") as f:
        json.dump(buf, fp=f)
    return

def create_eater(unix_path):
    from .gpu_eater import GPU_Eater
    proc = GPU_Eater(unix_path)
    proc.daemon = True
    proc.start()



class GpuHolder():
    def __init__(self, device) -> None:
        # try to communicate with gpu holder
        unix_path = os.path.expanduser(f'~/HmapTemp/GpuLock/GpuEater_{device}')
        try:
            self.client = UnixTcpClientP2P(unix_path, obj='str')
            success = self.client.send_and_wait_reply('link')
            print('already have a GpuHolder online')
        except:
            assert False
            print('creating GpuHolder')
            create_eater(unix_path)
            time.sleep(3)
            print('creating Finished')
            self.client = UnixTcpClientP2P(unix_path, obj='str')
            success = self.client.send_and_wait_reply('link')
        assert success == 'success'

    def __del__(self):
        if self.client is not None:
            self.client.send_and_wait_reply('offline')
            self.client.__del__()

    def need_gpu(self):
        ok = self.client.send_and_wait_reply('need_gpu')
        assert ok == 'ok'
    
    def giveup_gpu(self):
        ok = self.client.send_and_wait_reply('giveup_gpu')
        assert ok == 'ok'


class GpuShareUnit():
    flesh = True
    def __init__(self, which_gpu, lock_path=None, manual_gpu_ctl=True, gpu_party='', gpu_ensure_safe=False):
        self.device = which_gpu
        self.manual_gpu_ctl = True
        self.lock_path=lock_path
        self.gpu_party = gpu_party
        self.gpu_lock = None
        self.ensure_gpu_safe = gpu_ensure_safe
        self.pid_str = str(os.getpid())
        self.n_gpu_process_online = 1
        if self.ensure_gpu_safe:
            assert 'party0' in self.gpu_party; assert 'cuda' in self.gpu_party
            self.gpu_eater = GpuHolder(device=self.gpu_party)
        if gpu_party == 'off':
            self.manual_gpu_ctl = False
        # the default file lock path
        if self.lock_path is None: 
            self.lock_path = os.path.expanduser('~/HmapTemp/GpuLock')
        # create a folder if the path is invalid
        if not os.path.exists(self.lock_path): 
            os.makedirs(self.lock_path)
        # gpu party register file
        self.register_file = self.lock_path+'/lock_gpu_%s_%s.json'%(self.device, self.gpu_party)
        register(self.__del__)
        
    def __del__(self):
        if hasattr(self,'_deleted_'): 
            # avoid exit twice
            return
        else: 
            self._deleted_ = True     # avoid exit twice

        try:
            with FileLock(self.register_file+'.lock'):
                self.unregister_pid()
        except:
            pass

        try: self.gpu_lock.__exit__(None,None,None)
        except:pass

    def __enter__(self):
        self.get_gpu_lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_gpu_lock()

    def get_gpu_lock(self):
        if self.manual_gpu_ctl:
            print('Waiting for GPU %s %s...'%(self.device, self.gpu_party), end='', flush=True)
            with FileLock(self.register_file+'.lock'):
                self.n_gpu_process_online = self.register_pid()
            fp = self.lock_path+'/gpu_lock_%s_%s'%(self.device, self.gpu_party)
            self.gpu_lock = FileLock(fp+'.lock')
            self.gpu_lock.__enter__()
            if self.ensure_gpu_safe: self.gpu_eater.need_gpu()
            print('Get GPU, currently shared with %d process!'%self.n_gpu_process_online)
        return

    def release_gpu_lock(self):
        if self.manual_gpu_ctl:
            # if self.n_gpu_process_online > 1: 
            torch.cuda.empty_cache()
            if self.ensure_gpu_safe: self.gpu_eater.giveup_gpu()
            self.gpu_lock.__exit__(None,None,None)
            # else:
                # print('GPU not shared')
        return
    
    def register_pid(self):
        all_pids = read_json(self.register_file)
        need_write = False

        # check all pid alive occasionally
        if GpuShareUnit.flesh or random.random() < 0.05:
            for pid in list(all_pids.keys()):
                if not pid_exist(pid):
                    all_pids.pop(pid); print('removing dead item', pid)
                    need_write = True
            GpuShareUnit.flesh = False

        # add entry if not exist
        if self.pid_str not in all_pids:
            all_pids[self.pid_str] = {}
            need_write = True

        # write back if needed
        if need_write: write_json(self.register_file, all_pids)

        return len(all_pids)

    def unregister_pid(self):
        all_pids = read_json(self.register_file)

        # check all pid alive
        for pid in list(all_pids.keys()):
            if not pid_exist(pid):
                all_pids.pop(pid); print('removing dead item', pid)

        try:
            all_pids.pop(self.pid_str)
        except:
            pass

        # write back if needed
        write_json(self.register_file, all_pids)



