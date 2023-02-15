def validate_path():
    import os, sys
    dir_name = os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)

if __name__ == '__main__':
    validate_path()
    
from multiprocessing import Process
from UTIL.network import UnixTcpServerMultiClient
import os, time, re, torch
import threading

def check_devices_mem():
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi"' + 
        ' --query-gpu=memory.total,memory.used' +
        ' --format=csv,nounits,noheader'
    ).read().strip().split("\n")
    divices_mem_info = [x.split(',') for x in devices_info]
    divices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if divices is None:
        return divices_mem_info
    else:
        device_list = []
        for i in [int(x) for x in divices.split(',')]:
            device_list.append(divices_mem_info[i])
        return device_list


def occupy_device_mem(cuda_device, mem_info, free=1024):
    total, used = int(mem_info[0]), int(mem_info[1])
    block_mem = total - used - free
    if block_mem > 0:
        print('Occupy device_{}\'s mem ...'.format(cuda_device))
        x = torch.zeros(
            (256, 1024, block_mem),
            dtype=torch.float32,
            device='cuda:{}'.format(cuda_device)
        )
        del x
        print('Occupy device_{}\'s mem finished'.format(cuda_device))
    else:
        print('Device_{}\'s out of memory'.format(cuda_device))


def occupy_gpus_mem(free=4096):
    for i, mem_info in enumerate(check_devices_mem()):
        occupy_device_mem(i, mem_info, free)
    print('Occupy all device\'s mem finished')


class GPU_Eater(Process):
    def __init__(self, unix_path, party):
        super(GPU_Eater, self).__init__()
        self.unix_path = unix_path
        self.server = None
        self.party = party
        match_res = re.match(pattern=r'cuda(.)_party(.)', string=party)
        cudax, self.party_index = match_res[1], match_res[2]
        assert self.party_index == '0'
        self.device = f'cuda:{cudax}'
        cudax_int = int(cudax)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cudax
        self.previous_req = time.time()

    def __del__(self):
        if self.server is not None:
            self.server.close()
        self.terminate()
        
    def run_timer(self):
        while True:
            time.sleep(60)
            delta_time = time.time() - self.previous_req
            print(f'inactive for {delta_time} seconds')
            if delta_time > 3600:
                self.__del__()
                break

    def release_gpu(self):
        torch.cuda.empty_cache()
        pass

    def hold_gpu(self):
        occupy_gpus_mem(free=2048)
        pass

    def on_receive_data(self, data):
        print('data incoming')
        if data == 'link':
            self.previous_req = time.time()
            reply = 'success'
        elif data == 'need_gpu':
            self.release_gpu()
            self.previous_req = time.time()
            reply = 'ok'
        elif data == 'giveup_gpu':
            self.hold_gpu()
            self.previous_req = time.time()
            reply = 'ok'
        elif data == 'offline':
            self.previous_req = time.time()
            reply = 'ok'
        else:
            assert False
        print(data)
        return reply

    def run(self):
        print('started')
        try: os.unlink(self.unix_path)
        except: pass
        t = threading.Thread(target=self.run_timer)
        t.daemon = True
        t.start()
        self.server = UnixTcpServerMultiClient(self.unix_path, obj='str')
        self.server.on_receive_data = lambda data: self.on_receive_data(data)
        self.server.be_online()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gpu_party')
    parser.add_argument('--party', type=str)
    args = parser.parse_args()
    party = args.party
    unix_path = os.path.expanduser(f'~/HmapTemp/GpuLock/GpuEater_{party}')
    o = GPU_Eater(unix_path, party)
    o.run()