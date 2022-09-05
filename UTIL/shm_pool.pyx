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
import time, pickle, platform, setproctitle, numpy
from multiprocessing import Process, RawValue, Semaphore
from multiprocessing import shared_memory
from ctypes import c_bool, c_uint32
from .hmp_daemon import kill_process_and_its_children
SHARE_BUF_SIZE = 10485760 # 10 MB for parameter buffer
REGULAR_BUF_SIZE = 500000 # The non-numpy content max buffer size

def print_red(*kw,**kargs):
    print("\033[1;31m",*kw,"\033[0m",**kargs)

def print_green(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)

# optimize share mem IO for numpy ndarray
class ndarray_indicator():
    def __init__(self, shape, dtype, shm_start, shm_end):
        self.shape = shape
        self.dtype = dtype
        self.shm_start = shm_start
        self.shm_end = shm_end
        self.count = (self.shm_end-self.shm_start)//self.dtype.itemsize

# optimize share mem IO for numpy ndarray
def convert_ndarray(numpy_ndarray, shm_pointer, shm):
    nbyte = numpy_ndarray.nbytes
    shape = numpy_ndarray.shape
    dtype = numpy_ndarray.dtype
    assert shm_pointer+nbyte < SHARE_BUF_SIZE, ('share memory overflow, need at least %d, yet only have %d'%(shm_pointer+nbyte, SHARE_BUF_SIZE))
    shm_array_object = numpy.ndarray(shape, dtype=dtype, buffer=shm[shm_pointer:shm_pointer+nbyte])
    shm_array_object[:] = numpy_ndarray[:]
    NID = ndarray_indicator(shape, dtype, shm_pointer, shm_pointer+nbyte)
    shm_pointer = shm_pointer+nbyte
    return NID, shm_pointer

# optimize share mem IO for numpy ndarray
def deepin(obj, shm, shm_pointer):
    if isinstance(obj, list): iterator_ = enumerate(obj)
    elif isinstance(obj, dict): iterator_ = obj.items()
    else: 
        assert not isinstance(obj, tuple)
        return shm_pointer
    for k, v in iterator_:
        if isinstance(v, (list,dict)):
            shm_pointer = deepin(v, shm, shm_pointer)
        elif isinstance(v, tuple):
            item2 = list(v)
            shm_pointer = deepin(item2, shm, shm_pointer)
            obj[k] = tuple(item2)
        elif isinstance(v, numpy.ndarray):
            if v.dtype == 'object':
                print('dtype is object, which is low efficient and may cause error!')
            NID, shm_pointer = convert_ndarray(v, shm_pointer, shm)
            obj[k] = NID
        else:
            continue
    return shm_pointer

# optimize share mem IO for numpy ndarray
def opti_numpy_object(obj, shm):
    shm_pointer = REGULAR_BUF_SIZE 
    shm_pointer = deepin(obj, shm, shm_pointer=REGULAR_BUF_SIZE)
    return obj

# optimize share mem IO for numpy ndarray
def reverse_deepin(obj, shm):
    if isinstance(obj, list): iterator_ = enumerate(obj)
    elif isinstance(obj, dict): iterator_ = obj.items()
    else: return
    for k, v in iterator_:
        if isinstance(v, (list,dict)):
            reverse_deepin(v, shm)
        elif isinstance(v, tuple):
            item2 = list(v)
            reverse_deepin(item2, shm)
            obj[k] = tuple(item2)
        elif isinstance(v, ndarray_indicator):
            obj[k] = numpy.frombuffer(shm, dtype=v.dtype, offset=v.shm_start, count=v.count).reshape(v.shape)
    return

# optimize share mem IO for numpy ndarray
def reverse_opti_numpy_object(obj, shm):
    reverse_deepin(obj, shm)
    return obj







class SuperProc(Process):
    """
        Child process worker (efficient distributed worker)
    """
    # initialize traffic IO
    def __init__(self, index, smib, smiobli, smtl, buf_size_limit, base_seed, sem_push, sem_pull):
        super(SuperProc, self).__init__()
        self.shared_memory = smib
        self.shared_memory_io_buffer = smib.buf
        self.shared_memory_io_buffer_len_indicator = smiobli
        self.shared_memory_traffic_light = smtl
        self.buf_size_limit = buf_size_limit
        self.local_seed = index + base_seed
        self.index = index
        self.sem_push = sem_push
        self.sem_pull = sem_pull
        self.target_tracker = []
        
    # on parent exit
    def __del__(self):
        if hasattr(self,'_deleted_'): return    # avoid exit twice
        else: self._deleted_ = True     # avoid exit twice
        self.shared_memory.close()
        for target_name in self.target_tracker: 
            setattr(self, target_name, None)    # GC by clearing the pointer.
        # force terminate all child process
        try: kill_process_and_its_children(self)
        except Exception as e: print_red('[shm_pool]: error occur when kill_process_and_its_children:\n', e)

    # add any class level objects
    def automatic_generation(self, name, gen_fn, *arg):
        setattr(self, name, gen_fn(*arg))

    # add any class level objects
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

    # execute any class method, return the results
    def execute_target(self, recv_args):
        res_list = [None] * len(recv_args)
        for i, recv_arg in enumerate(recv_args):
            name, dowhat, arg = recv_arg
            if dowhat == 'None':
                continue
            if arg is None:              
                res = getattr(getattr(self, name), dowhat)()
            elif isinstance(arg, tuple): 
                res = getattr(getattr(self, name), dowhat)(*arg)
            else:                        
                res = getattr(getattr(self, name), dowhat)(arg)
            res_list[i] = res
        return res_list

    # inf loop, controlled / blocked by semaphore
    def run(self):
        import numpy, platform
        numpy.random.seed(self.local_seed)
        # set top process title
        setproctitle.setproctitle('HmapShmPoolWorker_%d'%self.index)
        try:
            while True:
                recv_args = self._recv_squence() # block and wait incoming req
                if not isinstance(recv_args, list): # not list object, switch to helper channel
                    if recv_args == 0: 
                        self._set_done() # must be put between _recv_squence()
                        self.add_targets(self._recv_squence())
                        self._set_done()
                    elif recv_args == -1: # termination signal
                        self._set_done() # ('superProc exiting')
                        break # terminate
                    else: # exception
                        assert False
                    continue
                # if list, execute target                
                result = self.execute_target(recv_args)
                # return the results (self._set_done() is called inside)
                self._send_squence(result)

        except KeyboardInterrupt:
            # ('child KeyboardInterrupt: close unlink')
            self.__del__()

        # ('child end: close unlink')
        self.__del__()


    # block and wait incoming req 
    def _recv_squence(self): 
        self.sem_push.acquire()
        assert self.shared_memory_traffic_light.value == True
        bufLen = self.shared_memory_io_buffer_len_indicator.value
        recv_args = pickle.loads(self.shared_memory_io_buffer[:bufLen])
        recv_args = reverse_opti_numpy_object(recv_args, shm=self.shared_memory_io_buffer)
        return recv_args

    # return results
    def _send_squence(self, send_obj):
        assert self.shared_memory_traffic_light.value == True
        # second prepare parameter
        send_obj = opti_numpy_object(send_obj, shm=self.shared_memory_io_buffer)
        picked_obj = pickle.dumps(send_obj, protocol=pickle.HIGHEST_PROTOCOL)
        lenOfObj = len(picked_obj)
        assert lenOfObj <= REGULAR_BUF_SIZE, ('The non-numpy content size > 0.5MB, please check!', lenOfObj)
        self.shared_memory_io_buffer_len_indicator.value = lenOfObj
        self.shared_memory_io_buffer[:lenOfObj] = picked_obj
        # then light up the work flag, turn off the processed flag
        self.shared_memory_traffic_light.value = False  # CORE! the job is done, waiting for next one
        self.sem_pull.release()

    # set traffic IO flag
    def _set_done(self):
        self.shared_memory_traffic_light.value = False  # CORE! the job is done, waiting for next one
        self.sem_pull.release()






class SmartPool(object):
    """
        Main parallel runner / coodinator
    """
    # setup and spawn workers
    def __init__(self, proc_num, fold, base_seed=None):
        self.proc_num = proc_num
        self.task_fold = fold
        self.base_seed = int(numpy.random.rand()*1e5) if base_seed is None else base_seed
        self.buf_size_limit = SHARE_BUF_SIZE # 10 MB for parameter buffer
        print_green('Linux multi-env using share memory')
        setproctitle.setproctitle('HmapRootProcess')
        self.shared_memory_io_buffer_handle = [shared_memory.SharedMemory(create=True, size=SHARE_BUF_SIZE) for _ in range(proc_num)]
        self.shared_memory_io_buffer_len_indicator = [RawValue(c_uint32, 0) for _ in range(proc_num)]
        self.shared_memory_traffic_light = [RawValue(c_bool, False) for _ in range(proc_num)] # time to work flag
        self.last_time_response_handled = [True for _ in range(proc_num)] # time to work flag
        self.semaphore_push = [Semaphore(value=0) for _ in range(proc_num)] # time to work flag
        self.semaphore_pull = Semaphore(value=0) # time to work flag
        self.proc_pool = [SuperProc(cnt, smib, smiobli, smtl, SHARE_BUF_SIZE, self.base_seed, 
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
    
    # add class level targets in each worker
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

    # run class method in each worker
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

    # low-level send
    def _send_squence(self, send_obj, target_proc):
        assert self.last_time_response_handled[target_proc] == True
        send_obj = opti_numpy_object(send_obj, shm=self.shared_memory_io_buffer[target_proc])
        picked_obj = pickle.dumps(send_obj, protocol=pickle.HIGHEST_PROTOCOL)
        lenOfObj = len(picked_obj)
        assert lenOfObj <= REGULAR_BUF_SIZE, ('The non-numpy content size > 0.5MB, please check!', lenOfObj)
        self.shared_memory_io_buffer_len_indicator[target_proc].value = lenOfObj
        self.shared_memory_io_buffer[target_proc][:lenOfObj] = picked_obj
        self.last_time_response_handled[target_proc] = False  # then light up the work flag, turn off the processed flag
        self.shared_memory_traffic_light[target_proc].value = True  

    # low-level recv
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


    # low-level wait
    def _wait_done(self, target_proc):   # used only in add_target
        self.semaphore_pull.acquire()
        self.last_time_response_handled[target_proc] = True

    # let all workers know about incomming req
    def notify_all_children(self):
        for j in range(self.proc_num):  
            self.semaphore_push[j].release()  # notify all child process

    # exit and clean up carefully
    def party_over(self):
        self.__del__()

    # exit and clean up carefully
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

        # kill shm_pool's process tree
        print_red('[shm_pool]: kill_process_and_its_children(proc)')
        for proc in self.proc_pool: 
            try: kill_process_and_its_children(proc)
            except Exception as e: print_red('[shm_pool]: error occur when kill_process_and_its_children:\n', e)
            


        print_green('[shm_pool]: __del__ finish')
        self.terminated = True

# To compat Windows, redirect to pipe solution
if not platform.system()=="Linux":  
    from UTIL.win_pool import SmartPool
    