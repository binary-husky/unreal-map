"""
    Author: Fu Qingxu,CASIA
    Description: Efficient parallel execting tool, 
        it resembles Ray but:
            1.optimized for single machine using shared memory
            2.optimized for numpy ndarray
            3.use semaphore for IPC
            4.faster!
    Note: 
        SHARE_BUF_SIZE: shared memory size, 10MB per process
"""
import time, pickle, platform, datetime
import numpy as np
from multiprocessing import Process, RawValue, Semaphore
from multiprocessing import shared_memory
from ctypes import c_char, c_uint16, c_bool, c_uint32, c_byte
from time import sleep as _sleep
from .hmp_daemon import kill_process_and_its_children
SHARE_BUF_SIZE = 10485760
def print_red(*kw,**kargs):
    print("\033[1;31m",*kw,"\033[0m",**kargs)
def print_green(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)


"""
    Part 1: optimized for numpy ndarray
"""
class ndarray_indicator():
    def __init__(self, shape, dtype, shm_start, shm_end):
        self.shape = shape
        self.dtype = dtype
        self.shm_start = shm_start
        self.shm_end = shm_end
        self.count = (self.shm_end-self.shm_start)//self.dtype.itemsize

def convert_ndarray(numpy_ndarray, shm_pointer, shm):
    nbyte = numpy_ndarray.nbytes
    shape = numpy_ndarray.shape
    dtype = numpy_ndarray.dtype
    assert shm_pointer+nbyte < SHARE_BUF_SIZE, ('share memory overflow, need at least %d, yet only have %d'%(shm_pointer+nbyte, SHARE_BUF_SIZE))
    shm_array_object = np.ndarray(shape, dtype=dtype, buffer=shm[shm_pointer:shm_pointer+nbyte])
    shm_array_object[:] = numpy_ndarray[:]
    NID = ndarray_indicator(shape, dtype, shm_pointer, shm_pointer+nbyte)
    shm_pointer = shm_pointer+nbyte
    return NID, shm_pointer

def opti_numpy_object(obj, shm):
    shm_pointer = 500000 # 在0.5MB的位置  10 MB for parameter buffer
    def deepin(obj, shm_pointer):
        if isinstance(obj, list):
            for i_th in range(len(obj)):
                if isinstance(obj[i_th], list):
                    shm_pointer = deepin(obj[i_th], shm_pointer)
                elif isinstance(obj[i_th], tuple):
                    item2 = list(obj[i_th])
                    shm_pointer = deepin(item2, shm_pointer)
                    obj[i_th] = tuple(item2)
                elif isinstance(obj[i_th], np.ndarray):
                    if obj[i_th].dtype == 'object':
                        print('dtype is object, which is low efficient and may cause error!')
                    NID, shm_pointer = convert_ndarray(obj[i_th], shm_pointer, shm)
                    obj[i_th] = NID
                else:
                    continue
        return shm_pointer
    shm_pointer = deepin(obj, shm_pointer)
    return obj
        
def reverse_opti_numpy_object(obj, shm):
    def reverse_deepin(obj):
        if isinstance(obj, list):
            for i_th in range(len(obj)):
                if isinstance(obj[i_th], list):
                    reverse_deepin(obj[i_th])
                elif isinstance(obj[i_th], tuple):
                    item2 = list(obj[i_th])
                    reverse_deepin(item2)
                    obj[i_th] = tuple(item2)
                elif isinstance(obj[i_th], ndarray_indicator):
                    NID = obj[i_th]
                    obj[i_th] = np.frombuffer(shm, dtype=NID.dtype, offset=NID.shm_start, count=NID.count).reshape(NID.shape)
                else:
                    continue
    reverse_deepin(obj)
    return obj






"""
    Part 2: child process worker
"""
class SuperProc(Process):
    def __init__(self, index, smib, smiobli, smtl, buf_size_limit, base_seed, sem_push, sem_pull):
        super(SuperProc, self).__init__()
        self.shared_memory = smib
        self.shared_memory_io_buffer = smib.buf
        self.shared_memory_io_buffer_len_indicator = smiobli
        self.shared_memory_traffic_light = smtl                 # time to work flag
        self.buf_size_limit = buf_size_limit
        self.local_seed = index + base_seed
        self.index = index
        self.sem_push = sem_push
        self.sem_pull = sem_pull
        self.target_tracker = []

    def __del__(self):
        if hasattr(self,'_deleted_'): return    # avoid exit twice
        else: self._deleted_ = True     # avoid exit twice
        self.shared_memory.close()
        for target_name in self.target_tracker: 
            setattr(self, target_name, None)    # GC by clearing the pointer.
        return

    def automatic_generation(self, name, gen_fn, *arg):
        setattr(self, name, gen_fn(*arg))

    def automatic_execution(self, name, dowhat, *arg):
        return getattr(getattr(self, name), dowhat)(*arg)

    def add_targets(self, new_tarprepare_args):
        for new_target_arg in new_tarprepare_args:
            name, gen_fn, arg = new_target_arg
            if name not in self.target_tracker: self.target_tracker.append(name)
            if arg is None:
                self.automatic_generation(name, gen_fn)
            elif isinstance(arg, tuple): 
                self.automatic_generation(name, gen_fn, *arg)
            else:                        
                self.automatic_generation(name, gen_fn, arg)


    def execute_target(self, recv_args):
        res_list = [None] * len(recv_args)
        for i, recv_arg in enumerate(recv_args):
            name, dowhat, arg = recv_arg
            if dowhat == 'None':
                continue
            if arg is None:              
                # res = self.automatic_execution(name, dowhat)
                res = getattr(getattr(self, name), dowhat)()
            elif isinstance(arg, tuple): 
                # res = self.automatic_execution(name, dowhat, *arg)
                res = getattr(getattr(self, name), dowhat)(*arg)
            else:                        
                # res = self.automatic_execution(name, dowhat, arg)
                res = getattr(getattr(self, name), dowhat)(arg)
            res_list[i] = res
        return res_list


    def run(self):
        import numpy, platform
        numpy.random.seed(self.local_seed)
        # linux uses fork, but windows does not, reload config for windows
        # if not platform.system()=="Linux":  child_process_load_config()   # disable, move to main.py
        try:
            while True:
                recv_args = self._recv_squence() # <<stage 1>>
                if not isinstance(recv_args, list): # not list object, switch to helper channel
                    if recv_args == 0: 
                        self._set_done() # must be put between _recv_squence()
                        self.add_targets(self._recv_squence())
                        self._set_done()
                    elif recv_args == -1: 
                        self._set_done() # print('superProc exiting')
                        break # terminate
                    else: 
                        assert False
                    continue
                result = self.execute_target(recv_args)
                self._send_squence(result)   # self._set_done() inside
        except KeyboardInterrupt:
            # print('child KeyboardInterrupt: close unlink')
            self.__del__()

        # print('child end: close unlink')
        self.__del__()



    def _recv_squence(self): 
        self.sem_push.acquire()
        assert self.shared_memory_traffic_light.value == True
        bufLen = self.shared_memory_io_buffer_len_indicator.value
        recv_args = pickle.loads(self.shared_memory_io_buffer[:bufLen])
        recv_args = reverse_opti_numpy_object(recv_args, shm=self.shared_memory_io_buffer)
        return recv_args

    def _send_squence(self, send_obj):
        assert self.shared_memory_traffic_light.value == True
        # second prepare parameter
        send_obj = opti_numpy_object(send_obj, shm=self.shared_memory_io_buffer)
        picked_obj = pickle.dumps(send_obj, protocol=pickle.HIGHEST_PROTOCOL)
        lenOfObj = len(picked_obj)
        assert lenOfObj <= 500000, ('The non-numpy content size > 0.5MB, please check!', lenOfObj)
        self.shared_memory_io_buffer_len_indicator.value = lenOfObj
        self.shared_memory_io_buffer[:lenOfObj] = picked_obj
        # then light up the work flag, turn off the processed flag
        self.shared_memory_traffic_light.value = False  # CORE! the job is done, waiting for next one
        self.sem_pull.release()

    def _set_done(self):
        self.shared_memory_traffic_light.value = False  # CORE! the job is done, waiting for next one
        self.sem_pull.release()




"""
    Part 3: main parallel processer
"""

class SmartPool(object):
    def __init__(self, proc_num, fold, base_seed=None):
        self.proc_num = proc_num
        self.task_fold = fold
        self.base_seed = int(np.random.rand()*1e5) if base_seed is None else base_seed
        # define shared object size
        CC_DEF_SHARED_OBJ_BUF_SIZE = SHARE_BUF_SIZE # 10 MB for parameter buffer
        self.buf_size_limit = CC_DEF_SHARED_OBJ_BUF_SIZE
        print('Multi-env using share memory')
        self.shared_memory_io_buffer_handle = [shared_memory.SharedMemory(create=True, size=SHARE_BUF_SIZE) for _ in range(proc_num)]
        self.shared_memory_io_buffer_len_indicator = [RawValue(c_uint32, 0) for _ in range(proc_num)]
        self.shared_memory_traffic_light = [RawValue(c_bool, False) for _ in range(proc_num)] # time to work flag
        self.last_time_response_handled = [True for _ in range(proc_num)] # time to work flag
        self.semaphore_push = [Semaphore(value=0) for _ in range(proc_num)] # time to work flag
        self.semaphore_pull = Semaphore(value=0) # time to work flag
        self.proc_pool = [SuperProc(cnt, smib, smiobli, smtl, CC_DEF_SHARED_OBJ_BUF_SIZE, self.base_seed, 
                                    sem_push, self.semaphore_pull)
                            for cnt, smib, smiobli, smtl, sem_push in 
                            zip(range(proc_num),
                                self.shared_memory_io_buffer_handle, self.shared_memory_io_buffer_len_indicator, 
                                self.shared_memory_traffic_light, self.semaphore_push
                            )]
        self.shared_memory_io_buffer = [shm.buf for shm in self.shared_memory_io_buffer_handle]
        self.t_profile = 0
        for proc in self.proc_pool:
            # proc.daemon = True
            proc.start()

    def _send_squence(self, send_obj, target_proc):
        assert self.last_time_response_handled[target_proc] == True
        send_obj = opti_numpy_object(send_obj, shm=self.shared_memory_io_buffer[target_proc])
        picked_obj = pickle.dumps(send_obj, protocol=pickle.HIGHEST_PROTOCOL)
        lenOfObj = len(picked_obj)
        assert lenOfObj <= 500000, ('The non-numpy content size > 0.5MB, please check!', lenOfObj)
        self.shared_memory_io_buffer_len_indicator[target_proc].value = lenOfObj
        self.shared_memory_io_buffer[target_proc][:lenOfObj] = picked_obj
        self.last_time_response_handled[target_proc] = False  # then light up the work flag, turn off the processed flag
        self.shared_memory_traffic_light[target_proc].value = True  

    def _recv_squence_all(self):
        res_sort = [None for j in range(self.proc_num*self.task_fold)]
        not_ready = [True for j in range(self.proc_num)]
        n_acq = 0
        ready_n = 0
        while True:
            self.semaphore_pull.acquire()   # wait child process and OS coordination, it will take a moment
            n_acq += 1
            for target_proc, not_r in enumerate(not_ready):
                if not not_r: continue  # finish already
                if self.shared_memory_traffic_light[target_proc].value: continue  # not ready
                bufLen = self.shared_memory_io_buffer_len_indicator[target_proc].value
                recv_obj = pickle.loads(self.shared_memory_io_buffer[target_proc][:bufLen])
                recv_obj = reverse_opti_numpy_object(recv_obj, shm=self.shared_memory_io_buffer[target_proc])

                self.last_time_response_handled[target_proc] = True
                res_sort[target_proc*self.task_fold: (target_proc+1)*self.task_fold] = recv_obj
                not_ready[target_proc] = False
                ready_n += 1
            
            if ready_n == self.proc_num:
                break

        for _ in range(self.proc_num-n_acq):
            self.semaphore_pull.acquire()  # clear semaphore_pull
        return res_sort
    

    def add_target(self, name, lam, args_list=None):
        lam_list = None
        if isinstance(lam, list): lam_list = lam
        # send command for workers to wait appending new target
        for j in range(self.proc_num):
            self._send_squence(send_obj=0, target_proc=j)
        self.notify_all_children()
        for j in range(self.proc_num): self._wait_done(j)

        for j in range(self.proc_num):
            tuple_list_to_be_send = []
            for i in range(self.task_fold):
                name_fold = name + str(i)
                args = None if args_list is None else args_list[i + j*self.task_fold]
                if lam_list is not None: lam = lam_list[i + j*self.task_fold]
                tuple_list_to_be_send.append((name_fold, lam, args))
            self._send_squence(send_obj=tuple_list_to_be_send, target_proc=j)
        self.notify_all_children()
        for j in range(self.proc_num): self._wait_done(j)

    def _wait_done(self, target_proc):   # used only in add_target
        self.semaphore_pull.acquire()
        self.last_time_response_handled[target_proc] = True

    def exec_target(self, name, dowhat, args_list = None, index_list = None):
        t_sum = 0
        if index_list is not None:
            for j in range(self.proc_num):
                tuple_list_to_be_send = []
                for i in range(self.task_fold):
                    n_thread = i + j*self.task_fold
                    name_fold = name + str(i)
                    if n_thread in index_list:
                        args = None if args_list is None else args_list[index_list.index(n_thread)]
                        tuple_list_to_be_send.append((name_fold, dowhat, args))
                    else:
                        tuple_list_to_be_send.append((name_fold, 'None', 'None'))
                self._send_squence(send_obj=tuple_list_to_be_send, target_proc=j)
                self.semaphore_push[j].release()

        else: # if index_list is  None:
            for j in range(self.proc_num):
                tuple_list_to_be_send = []
                for i in range(self.task_fold):
                    name_fold = name + str(i)
                    args = None if args_list is None else args_list[i + j*self.task_fold]
                    tuple_list_to_be_send.append((name_fold, dowhat, args))
                self._send_squence(send_obj=tuple_list_to_be_send, target_proc=j)
                self.semaphore_push[j].release()
        res_sort = self._recv_squence_all()
        return res_sort

    def notify_all_children(self):
        for j in range(self.proc_num):  
            self.semaphore_push[j].release()  # notify all child process

    def party_over(self):
        self.__del__()

    def __del__(self):
        print('[shm_pool]: executing superpool del')
        # traceback.print_exc()
        if hasattr(self, 'terminated'): 
            print_red('[shm_pool]: already terminated, skipping ~')
            return

        try:
            for i in range(self.proc_num): self._send_squence(send_obj=-1, target_proc=i)
            self.notify_all_children()
            print('[shm_pool]: self.notify_all_children()')
        except: pass

        print('[shm_pool]: shm.close(); shm.unlink()')
        for shm in self.shared_memory_io_buffer_handle:
            try: shm.close(); shm.unlink()
            except: pass

        N_SEC_WAIT = 2
        for i in range(N_SEC_WAIT):
            print_red('[shm_pool]: terminate in %d'%(N_SEC_WAIT-i));time.sleep(1)

        # 杀死shm_pool创建的所有子进程，以及子进程的孙进程
        print_red('[shm_pool]: kill_process_and_its_children(proc)')
        for proc in self.proc_pool: 
            try: kill_process_and_its_children(proc)
            except Exception as e: print_red('[shm_pool]: error occur when kill_process_and_its_children:\n', e)
            


        print_green('[shm_pool]: __del__ finish')
        self.terminated = True

# To compat Windows
if not platform.system()=="Linux":  
    from UTIL.win_pool import SmartPool
    