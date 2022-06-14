# cython: language_level=3
import numpy as np
from multiprocessing import Pipe
import time
import multiprocessing


class SuperProc(multiprocessing.Process):
    def __init__(self, pipe, pipeHelp, index, base_seed):
        super(SuperProc, self).__init__()
        self.p = pipe
        self.pH = pipeHelp
        self.local_seed = index + base_seed
        self.index = index


    def __del__(self):
        self.p.close()
        self.pH.close()

    def automatic_generation(self, name, gen_fn, *arg):
        setattr(self, name, gen_fn(*arg))

    def automatic_execution(self, name, dowhat, *arg):
        return getattr(getattr(self, name), dowhat)(*arg)

    def add_targets(self, new_target_args):
        for new_target_arg in new_target_args:
            name, gen_fn, arg = new_target_arg
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
            if arg is None:
                res = self.automatic_execution(name, dowhat)
            elif isinstance(arg, tuple):
                res = self.automatic_execution(name, dowhat, *arg)
            else:
                res = self.automatic_execution(name, dowhat, arg)
            res_list[i] = res
        return res_list

    def run(self):
        import numpy
        numpy.random.seed(self.local_seed)
        print('process worker %d started'%self.index)
        try:
            while True:
                recv_args = self.p.recv()
                if not isinstance(recv_args, list):  # not list object, switch to helper channel
                    if recv_args == 0:
                        self.add_targets(self.pH.recv())
                    elif recv_args == -1:
                        print('子进程退出')
                        break  # terminate
                    else:
                        assert False
                    continue
                result = self.execute_target(recv_args)
                self.p.send(result)
        except KeyboardInterrupt:
            self.__del__()
        self.__del__()




class SmartPool(object):
    def __init__(self, proc_num, fold, base_seed=None):


        self.proc_num = proc_num
        self.task_fold = fold
        self.thisSide, self.thatSide = zip(*[Pipe() for _ in range(proc_num)])
        self.thisSideHelp, self.thatSideHelp = zip(*[Pipe() for _ in range(proc_num)])
        self.base_seed = int(np.random.rand()*1e5) if base_seed is None else base_seed
        print('SmartPool base rand seed', self.base_seed)
        self.proc_pool = [SuperProc(pipe=p, pipeHelp=pH, index=cnt, base_seed=self.base_seed)
                          for p, pH, cnt in zip(self.thatSide, self.thatSideHelp, range(proc_num))]
        for proc in self.proc_pool:
            proc.daemon = True
            proc.start()
            time.sleep(0.001)
        # shut down
        for i in range(proc_num):
            self.thatSide[i].close()
            self.thatSideHelp[i].close()

    def __del__(self):
        try:
            if hasattr(self, 'terminated'):
                return
        except:
            return

        print(' kill ')
        try:
            for i in range(self.proc_num):
                self.thisSide[i].send(-1)    # switch to helper channel
                print('向子进程发出退出指令')
            time.sleep(3)
            self.terminated = True
            for i in range(self.proc_num):
                self.thisSide[i].close()
                self.thisSideHelp[i].close()
        except:
            pass

    # add an object of some class, initialize it proc_num=64 times, assigning them to proc_num/fold_num=16 python
    # processes
    def add_target(self, name, lam, args_list=None):
        lam_list = None
        if isinstance(lam, list): lam_list = lam
        for j in range(self.proc_num):
            tuple_list_to_be_send = []
            for i in range(self.task_fold):
                name_fold = name + str(i)
                args = None if args_list is None else args_list[i + j*self.task_fold]
                if lam_list is not None: lam = lam_list[i + j*self.task_fold]
                tuple_list_to_be_send.append((name_fold, lam, args))
            self.thisSide[j].send(0)    # switch to helper channel
            self.thisSideHelp[j].send(tuple_list_to_be_send)

    # if there is index, execute one, otherwise execute all
    def exec_target(self, name, dowhat, args_list = None, index_list = None):
        if index_list is None:
            for j in range(self.proc_num):
                tuple_list_to_be_send = []
                for i in range(self.task_fold):
                    name_fold = name + str(i)
                    args = None if args_list is None else args_list[i + j*self.task_fold]
                    tuple_list_to_be_send.append((name_fold, dowhat, args))
                self.thisSide[j].send(tuple_list_to_be_send)
            res_sort = []
            for j in range(self.proc_num):
                res_sort.extend(self.thisSide[j].recv())
            return res_sort
        else:
            tuple_List_List = [[None for _ in range(self.task_fold)] for _ in range(self.proc_num)]
            do_task_flag = [False for _ in range(self.proc_num)]
            do_task_fold = [[] for _ in range(self.proc_num)]
            result_recv_List_List = [[None for _ in range(self.task_fold)] for _ in range(self.proc_num)]
            # sort args
            for i, index in enumerate(index_list):
                which_proc = index // self.task_fold
                which_fold = index % self.task_fold
                name_fold = name + str(which_fold)
                args = None if args_list is None else args_list[i]
                tuple_List_List[which_proc][which_fold] = (name_fold, dowhat, args)
                do_task_flag[which_proc] = True
                
            # send args
            for which_proc in range(self.proc_num):
                tuple_send_buffer = []
                for which_fold, item in enumerate(tuple_List_List[which_proc]):
                    if item is None: continue
                    tuple_send_buffer.append(item)
                    do_task_fold[which_proc].append(which_fold)
                if do_task_flag[which_proc]:
                    assert len(tuple_send_buffer) > 0
                    self.thisSide[which_proc].send(tuple_send_buffer)

            # receive returns
            for which_proc in range(self.proc_num):
                if not do_task_flag[which_proc]:
                    continue
                recv_tmp = self.thisSide[which_proc].recv()
                for index, recv_item in enumerate(recv_tmp):
                    which_fold = do_task_fold[which_proc][index]
                    result_recv_List_List[which_proc][which_fold] = recv_item

            # sort returns
            res_sort = [None] * len(index_list)
            for i, index in enumerate(index_list):
                which_proc = index // self.task_fold
                which_fold = index % self.task_fold
                res_sort[i] = result_recv_List_List[which_proc][which_fold]
            return res_sort

