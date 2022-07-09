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

    def send(self, msg):
        self.socketx.send(bytes(msg, encoding='utf8'))
        
    def __del__(self):
        self.socketx.close()


class MyHttp(Process):
    def __init__(self, path_to_html):
        super(MyHttp, self).__init__()
        self.path_to_html = path_to_html

    def run(self):
        from flask import Flask
        app = Flask(__name__)
        @app.route("/")
        def hello():
            try:
                with open(self.path_to_html,'r') as f:
                    html = f.read()
            except:
                html = "no plot yet please wait"
            return html

        def find_free_port():
            from contextlib import closing
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return s.getsockname()[1]

        avail_port = find_free_port()
        app.run(port=avail_port)
        
class DrawProcess(Process):
    def __init__(self, pipe, draw_mode):
        super(DrawProcess, self).__init__()
        self.draw_mode = draw_mode

        import matplotlib.pyplot as plt
        
        if self.draw_mode == 'Web':
            self.gui_reflesh = lambda: time.sleep(1) # plt.pause(0.1)
        elif self.draw_mode == 'Native':
            self.gui_reflesh = lambda: plt.pause(0.1)
        else:
            assert False

        from config import GlobalConfig
        note = GlobalConfig.note
        if not os.path.exists('./VISUALIZE/%s'%note):
            os.makedirs('./VISUALIZE/%s'%note)

        if self.draw_mode == 'Web':
            my_http = MyHttp('./VISUALIZE/%s/html.html'%note)
            my_http.daemon = True
            my_http.start()

        self.p = pipe
        self.libs_family = {
            'rec_init': 'rec', 'rec': 'rec', 'rec_show': 'rec',
            'v2d_init': 'v2d', 'v2dx':'v2d', 'v2d_show': 'v2d','v2d_line_object':'v2d'
        }
        self.libs_init_fns = {
            'rec': self.rec_init_fn,
            'v2d': self.v2d_init_fn,
        }
        pass

    def run(self):
        try:
            while True:
                cmd_arrive = self.p.poll()
                if cmd_arrive:
                    cmd_str = self.p.recv_bytes()
                    self.process_cmd(cmd_str)
                else:
                    self.gui_reflesh()
        except KeyboardInterrupt:
            self.__del__()
        self.__del__()

    def __del__(self):
        self.p.close()

    def process_cmd(self, cmd_str):
        cmd_str = cmd_str.decode()
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
            print('绘图函数不能处理：', cmd)
            return None
        family_name = self.libs_family[func_name]
        if self.libs_init_fns[family_name] is not None:
            self.libs_init_fns[family_name]()
            self.libs_init_fns[family_name] = None
        return 'self.%s'%family_name

    # lib
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