import socket
import os
import time
import traceback
import numpy as np
from colorama import init
from multiprocessing import Process, Pipe
init()



def find_free_index(path):
    if not os.path.exists(path): os.makedirs(path)
    t = 0
    while True:
        if os.path.exists(path + '/mcom_buffer_%d.txt' % t) \
                or os.path.exists(path + '/mcom_buffer_%d____starting_session.txt' % t):
            t += 1
        else:
            return t


class tcp_client():
    def __init__(self, ip):
        TCP_IP, TCP_PORT = ip.split(':')
        TCP_PORT = int(TCP_PORT)
        BUFFER_SIZE = 10240
        self.socketx = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socketx.connect((TCP_IP, TCP_PORT))

        self.fixed_end = bytes('@end@', encoding='utf8') 

    def send(self, str_msg):
        msg = str_msg + '@end@'
        self.socketx.send(bytes(msg, encoding='utf8'))

    def send_bytes(self, b_msg):
        msg = b_msg + self.fixed_end
        self.socketx.send(msg)

    def close(self):
        self.socketx.close()

    def __del__(self):
        self.socketx.close()

        
class tcp_server():
    def __init__(self, ip_port):
        self.draw_cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.draw_cmd_socket.bind(ip_port)
        self.draw_cmd_socket.listen()
        self.handler = None
        self.queue = None
        self.buff = ['']
        
    def wait_connection(self):
        import threading
        self.sock, _ = self.draw_cmd_socket.accept()
        t = threading.Thread(target=self.listening_thread)
        t.start()

    def listening_thread(self):


        def handle_flag_breakdown():
            split_ = self.buff[-1].split('@end@')
            assert len(split_)==2
            self.buff[-1] = split_[0]
            self.buff.append('')
            self.buff[-1] = split_[1]
            return

        assert (self.handler is None)     or (self.queue is None)
        assert (self.handler is not None) or (self.queue is not None)

        while True:
            recvData = self.sock.recv(10240)
            recvData = str(recvData, encoding = "utf-8")
            ends_with_mark = recvData.endswith('@end@')
            split_res = recvData.split('@end@')
            assert len(split_res) != 0
            if len(split_res) == 1:
                # 说明没有终止符，直接将结果贴到buf最后一项
                self.buff[-1] = self.buff[-1] + split_res[0]
                if '@end@' in self.buff[-1]: 
                    handle_flag_breakdown()
            else:
                n_split = len(split_res)
                for i, r in enumerate(split_res):
                    self.buff[-1] = self.buff[-1] + r
                    if i == 0 and '@end@' in self.buff[-1]:
                        handle_flag_breakdown()
                    if i != n_split-1:
                        self.buff.append('')
                    else: # 最后一个
                        if r == '': continue
            buff_list = self.buff[:-1]
            self.buff = self.buff[-1:]
            if self.handler is not None: 
                self.handler(buff_list)
            if self.queue is not None: 
                self.queue.put(buff_list)

        return

    def set_handler(self, handler):
        self.handler = handler

    def get_queue(self):
        import queue
        self.queue = queue.Queue()
        return self.queue

    def recv(self):
        return

    # def send(self, str_msg):
    #     msg = '@start@' + str_msg + '@end@'
    #     self.sock.send(bytes(msg, encoding='utf8'))

    def send_bytes(self, b_msg):
        msg = self.fixed_L + b_msg + self.fixed_R
        self.sock.send(msg)

    def close(self):
        if hasattr(self, 'sock'): self.sock.close()
        if hasattr(self, 'draw_cmd_socket'): self.draw_cmd_socket.close()        

    def __del__(self):
        self.close()



        
class DrawProcess(Process):
    def __init__(self, draw_udp_port, draw_mode):
        super(DrawProcess, self).__init__()
        self.draw_mode = draw_mode
        self.draw_udp_port = draw_udp_port
        self.tcp_connection = tcp_server(self.draw_udp_port)


        return

    def init_matplot_lib(self):
        import matplotlib
        if self.draw_mode in ['Web', 'Img']:
            matplotlib.use('Agg') # set the backend before importing pyplot
            import matplotlib.pyplot as plt
            self.gui_reflesh = lambda: time.sleep(1) # plt.pause(0.1)
        elif self.draw_mode == 'Native':
            # matplotlib.use('Agg') # set the backend before importing pyplot
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            # canvas = plt.gca().figure.canvas
            self.gui_reflesh = lambda: plt.pause(0.2) #canvas.start_event_loop(0.1) # plt.pause(0.2) #time.sleep(0.2) #
        else:
            assert False
        from config import GlobalConfig
        logdir = GlobalConfig.logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if self.draw_mode == 'Web':
            self.avail_port = find_free_port()
            my_http = MyHttp('%s/html.html'%logdir, self.avail_port)
            my_http.daemon = True
            my_http.start()
        self.libs_family = {
            'rec_init': 'rec', 'rec': 'rec', 'rec_show': 'rec',
            'v2d_init': 'v2d', 'v2dx':'v2d', 'v2d_show': 'v2d', 'v2d_pop':'v2d',
            'v2d_line_object':'v2d', 'v2d_clear':'v2d', 'v2d_add_terrain': 'v2d',
        }
        self.libs_init_fns = {
            'rec': self.rec_init_fn,
            'v2d': self.v2d_init_fn,
        }

    def run(self):
        self.init_matplot_lib()
        try:
            # self.tcp_connection.set_handler(self.run_handler)
            from queue import Empty
            queue = self.tcp_connection.get_queue()
            # self.tcp_connection.set_handler(self.run_handler)
            self.tcp_connection.wait_connection() # after this, the queue begin to work
            while True:
                try: self.run_handler(queue.get(timeout=0.1))
                except Empty: self.gui_reflesh()


        except KeyboardInterrupt:
            self.__del__()
        self.__del__()

    def run_handler(self, buff_list):
        for buff in buff_list:
            self.process_cmd(buff)
            # print('成功处理指令:', buff)

    def __del__(self):
        self.tcp_connection.close()



    def process_cmd(self, cmd_str):
        if '>>' in cmd_str:
            cmd_str_ = cmd_str[2:].strip('\n')
            if ')' not in cmd_str_:
                cmd_str_ = cmd_str_+'()'
            prefix = self.get_cmd_lib(cmd_str_)
            if prefix is not None: 
                eval('%s.%s'%(prefix, cmd_str_))

    def get_cmd_lib(self, cmd):
        cmd_key = None
        func_name = cmd.split('(')[0]
        if func_name not in self.libs_family:
            print蓝('绘图函数不能处理：', cmd)
            return None
        family_name = self.libs_family[func_name]
        if self.libs_init_fns[family_name] is not None:
            self.libs_init_fns[family_name]()
            self.libs_init_fns[family_name] = None
        return 'self.%s'%family_name

    def rec_init_fn(self):
        from VISUALIZE.mcom_rec import rec_family
        self.rec = rec_family('r', self.draw_mode)

    def v2d_init_fn(self):
        from VISUALIZE.mcom_v2d import v2d_family
        self.v2d = v2d_family(self.draw_mode)




def get_files_to_read(base_path):
    starting_file_index = -1
    ending_file_index = -1
    pointer = 0
    while True:
        es = os.path.exists(base_path+'mcom_buffer_%d____starting_session.txt'%pointer)
        ee = os.path.exists(base_path+'mcom_buffer_%d.txt'%pointer)
        if (not es) and (not ee): break
        assert not (ee and es), ('?')
        if es: starting_file_index = pointer; ending_file_index = pointer
        if ee: ending_file_index = pointer
        pointer += 1
        assert pointer < 1e3
    assert starting_file_index>=0 and ending_file_index>=0, ('查找日志失败')

    file_path = []
    for i in range(starting_file_index, ending_file_index+1):
        if i==starting_file_index: file_path.append(base_path+'mcom_buffer_%d____starting_session.txt'%i)
        else: file_path.append(base_path+'mcom_buffer_%d.txt'%i)
        assert os.path.exists(file_path[0]), ('?')
    return file_path

def read_experiment(base_path):
    files_to_read = get_files_to_read(base_path)
    cmd_lines = []
    for file in files_to_read:
        f = open(file, 'r')
        lines = f.readlines()
        cmd_lines.extend(lines)
    dictionary = {}

    def rec(value,name): 
        if name not in dictionary:
            dictionary[name] = []
        dictionary[name].append(value)
        return

    for cmd_str in cmd_lines:
        if '>>' in cmd_str:
            cmd_str_ = cmd_str[2:].strip('\n')
            if not cmd_str_.startswith('rec('): continue
            eval('%s'%cmd_str_)
    return dictionary

# if __name__ == '__main__':

#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns
    
#     #设置风格、尺度
#     sns.set_style('darkgrid')
#     sns.set_context('paper')
    
#     x = np.linspace(0, 15, 31)
#     data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
#     df = pd.DataFrame(data).melt()
#     sns.lineplot(x="variable", y="value", data=df)
#     plt.show()

def stack_cutlong(arr_list):
    min_len = min([len(item) for item in arr_list])
    print([len(item) for item in arr_list],'\tselect:', min_len)
    return np.stack([item[:min_len] for item in arr_list])


def tsplot(ax, data, label, resize_x, **kw):
    print('警告resize_x=',resize_x)
    x = np.arange(data.shape[1])
    x = resize_x*x
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est, label=label, **kw)
    ax.margins(x=0)





if __name__ == '__main__':
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #设置风格、尺度
    sns.set_style('darkgrid')
    sns.set_context('paper')

    note_base = 'Baseline'

    party = {
        'Ours':{
            'path':[
                './F-EXE001-seed9995-零干扰无地形起伏-Ours-C13-Focus2-24epoch-5e-4/logger/',
            ]
        },

        'Ours+AA':{
            'path':[
                './G-EXE006-Ours+AA-seed9992-无干扰无地形起伏-C14-Focus2-24epoch-5e-4/logger/',
            ]
        },

        'NoFocus':{
            'path':[
                './G-EXE009-NoFocus-seed9995-无干扰无地形起伏-C11-24epoch-5e-4/logger/',
            ]
        },
    }

    for exp_name in party:
        for path in party[exp_name]['path']:
            party[exp_name]['readings'] = read_experiment(path)
            print('reading ...')


    def my_plot(data, key, ax, label, resize_x):
        tsplot(ax, data, label, resize_x)   # data: (n_experiment, n_timestep)
        ax.set_title(key)
        

    fig, ax2 = plt.subplots(sharey=True)
    
    main_key = 'reward'
    for exp_name in party:
        data_batch = []
        for path in party[exp_name]['path']:
            data = party[exp_name]['readings'][main_key]
            data_batch.append(data)
        data_batch = stack_cutlong(data_batch)
        my_plot(data_batch, main_key, ax=ax2, label=exp_name,resize_x=64)
        pass
    ax2.legend()
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
    print('read finish!')
    


'''

    import numpy as np; np.random.seed(1)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    x = np.linspace(0, 15, 31)
    data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    df = pd.DataFrame(data).melt()
    sns.lineplot(x="variable", y="value", data=df)
    plt.show()

'''