import socket
import os
import time
import traceback
import numpy as np
from ..UTIL.colorful import *
from colorama import init
from multiprocessing import Process, Pipe
init()
def find_free_port():
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

'''
    mcom core function: draw line, by assigning points one by one, 
    for example 
        uc.rec(100,'live loss valueX')
        uc.rec(0.1,'entropy agentX')
        uc.rec(99, 'live loss valueX')
        uc.rec(0.3,'entropy agentX')
        uc.rec(88, 'live loss valueX')
        uc.rec(0.5,'entropy agentX')
        uc.rec_show()
'''
class mcom():
    # 这段小程序的设计原则是 在任何情况下 不干扰主线程序的运行！

    # as a recording programme, the design principle is that,
    # Under No Circumstance should this program Interrupt the main program!
    def __init__(self, ip=None, port=None, path=None, digit=8, rapid_flush=True, draw_mode=False):
        # digit 默认8，可选4,16，越小程序负担越轻 (all is float, set valid digit number)
        # rapid_flush 当数据流不大时，及时倾倒文件缓存内容 (set 'False' if you'd like your SSD to survive longer)
        self.draw_mode = draw_mode
        if draw_mode in ['Web', 'Native', 'Img', 'pyqtgraph']:
            self.draw_process = True; print红('draw process active!')
            port = find_free_port()
            self.draw_udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.draw_udp_port = ('localhost', port)
            self.draw_proc = DrawProcess(self.draw_udp_port, draw_mode)
            self.draw_proc.daemon = True
            self.draw_proc.start()
        else:
            print亮红('draw process off! no plot will be done')
            self.draw_process = False
        self.port = port
        self.dst = (ip, port)
        self.path = path
        self.current_buffer_index = find_free_index(self.path)
        self.starting_file = self.path + '/mcom_buffer_%d____starting_session.txt' % (self.current_buffer_index)
        self.current_file_handle = open(self.starting_file, 'wb+')
        self.file_lines_cnt = 0
        self.file_max_lines = 30000  # limit file lines to avoid a very fucking large file
        self.digit = digit
        self.rapid_flush = rapid_flush
        self.flow_cnt = 0
        # if self.port is not None:
        #     self.socketx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     str_tmp = ">>@" + str(self.current_buffer_index) + "@" + os.path.abspath(self.starting_file)
        #     b_tmp = bytes(str_tmp, encoding='utf8')
        #     self.socketx.sendto(b_tmp, self.dst)
        #     with open(os.path.expanduser('~/trace.comv5.txt'), 'a+', encoding='utf8') as f:
        #         f.writelines(['\n', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), '  |  ',
        #                       os.path.abspath(self.starting_file), '\n'])


        print蓝('**************mcom service initialized**************')
        print蓝('use MATLAB to open live log file at:' + self.starting_file)

    '''
        mcom core function: send out/write raw bytes
    '''

    def send(self, data):
        # step 1: add to file
        self.file_lines_cnt += 1
        self.current_file_handle.write(data)
        if self.rapid_flush: self.current_file_handle.flush()
        elif self.flow_cnt>500:
            self.current_file_handle.flush()
            self.flow_cnt = 0
        # step 2: check whether the file is too large, if so move on to next file.
        if self.file_lines_cnt > self.file_max_lines:
            end_file_flag = (b'><EndFileFlag\n')
            self.current_file_handle.write(end_file_flag)
            self.current_file_handle.close()
            self.current_buffer_index += 1
            self.current_file_handle = open((self.path + '/mcom_buffer_%d.txt' % self.current_buffer_index), 'wb+')
            self.file_lines_cnt = 0

        if self.draw_process: 
            self.draw_udp_client.sendto(data, self.draw_udp_port)
        # # step 3: UDP send directly
        # UDP function removed because it's not convenient

    def rec_init(self, color='k'):
        str_tmp = '>>rec_init(\'%s\')\n' % color
        b_tmp = bytes(str_tmp, encoding='utf8')
        self.send(b_tmp)

    def rec_show(self):
        b_tmp = b'>>rec_show\n'
        self.send(b_tmp)

    def rec_end(self):
        self.send(b'>>rec_end\n')

    def rec_save(self):
        self.send(b'>>rec_save\n')

    def rec_end_hold(self):
        self.send(b'>>rec_end_hold\n')

    def rec_clear(self, name):
        str_tmp = '>>rec_clear("%s")\n' % (name)
        str_tmp = bytes(str_tmp, encoding='utf8')
        self.send(str_tmp)

    def rec(self, value, name):
        value = float(value)
        if self.digit == 16:
            str_tmp = '>>rec(%.16e,"%s")\n' % (value, name)
        elif self.digit == 8:
            str_tmp = '>>rec(%.8e,"%s")\n' % (value, name)
        elif self.digit == 4:
            str_tmp = '>>rec(%.4e,"%s")\n' % (value, name)

        str_tmp = bytes(str_tmp, encoding='utf8')
        self.send(str_tmp)

    def 发送虚幻4数据流(self, x, y, z, pitch, yaw, roll):
        x = float(x)
        y = float(y)
        z = float(z)
        pitch = float(pitch)
        yaw = float(yaw)
        roll = float(roll)
        str_tmp = 'UE4>>(\"agent#1\",%.6e,%.6e,%.6e,%.6e,%.6e,%.6e)\n' % (x, y, z, pitch, yaw, roll)
        str_tmp = bytes(str_tmp, encoding='utf8')
        self.send(str_tmp)

    def 发送虚幻4数据流_多智能体(self, x_, y_, z_, pitch_, yaw_, roll_):
        str_list = ['UE4>>']
        for x, y, z, pitch, yaw, roll in zip(x_, y_, z_, pitch_, yaw_, roll_):
            x = float(x)
            y = float(y)
            z = float(z)
            pitch = float(pitch)
            yaw = float(yaw)
            roll = float(roll)
            str_tmp = '(\"agent#1\",%.5e,%.5e,%.5e,%.5e,%.5e,%.5e)' % (x, y, z, pitch, yaw, roll)
            str_list.append(str_tmp)
            str_list.append(';')
        str_list.append('\n')

        cmd = ''.join(str_list)
        self.send(bytes(cmd, encoding='utf8'))

    def other_cmd(self, *args, **kargs):
        func_name = traceback.extract_stack()[-2][2]
        strlist = ['>>', func_name, '(']

        for _i_ in range(len(args)):
            if isinstance(args[_i_], np.ndarray):
                strlist = self._process_ndarray(args[_i_], strlist)
            else:
                strlist = self._process_scalar(args[_i_], strlist)
        if len(kargs)>0:
            for _key_ in kargs:
                if isinstance(kargs[_key_], np.ndarray):
                    strlist = self._process_ndarray(kargs[_key_], strlist, _key_)
                else:
                    strlist = self._process_scalar(kargs[_key_], strlist, _key_)

        if strlist[len(strlist) - 1] == "(": strlist.append(")\n")
        else: strlist[len(strlist) - 1] = ")\n" # 把逗号换成后括号
        self.send(bytes(''.join(strlist), encoding='utf8'))

    def _process_scalar(self, arg, strlist,key=None):
        if key is not None: strlist += '%s='%key
        if isinstance(arg, int):
            strlist.append("%d" % arg)
            strlist.append(",")
        elif isinstance(arg, float):
            if self.digit == 16:  strlist.append("%.16e" % arg)
            elif self.digit == 8: strlist.append("%.8e" % arg)
            elif self.digit == 4: strlist.append("%.4e" % arg)
            strlist.append(",")
        elif isinstance(arg, str):
            strlist.append("\'"); strlist.append(arg)
            strlist.append("\'"); strlist.append(",")
        elif isinstance(arg, list):
            strlist.append(str(arg))
            strlist.append(",")
        elif hasattr(arg, 'dtype') and np.issubdtype(arg.dtype, np.integer):
            strlist.append("%d" % arg)
            strlist.append(",")
        elif hasattr(arg, 'dtype') and np.issubdtype(arg.dtype, np.floating):
            if self.digit == 16:  strlist.append("%.16e" % arg)
            elif self.digit == 8: strlist.append("%.8e" % arg)
            elif self.digit == 4: strlist.append("%.4e" % arg)
            strlist.append(",")
        else:
            print('输入的参数类型不能处理',arg.__class__)
        return strlist


    def _process_ndarray(self, args, strlist, key=None):
        if args[0].ndim == 1:
            if key is not None: strlist += '%s='%key
            sub_list = ["["] + ["%.3e " % t for t in args[0]] + ["]"]
            strlist += sub_list
            strlist.append(",")
        elif args[0].ndim == 2:
            print红('mcom：输入数组的维度大于1维，目前处理不了。')
        else:
            print红('mcom：输入数组的维度大于2维，目前处理不了。')
        return strlist



    exec('def plot(self,*args):\n  self.other_cmd(*args)\n')
    exec('def figure(self,*args):\n  self.other_cmd(*args)\n')
    exec('def hold(self,*args):\n  self.other_cmd(*args)\n')
    exec('def box(self,*args):\n  self.other_cmd(*args)\n')
    exec('def pause(self,*args):\n  self.other_cmd(*args)\n')
    exec('def clf(self,*args):\n  self.other_cmd(*args)\n')
    exec('def xlim(self,*args):\n  self.other_cmd(*args)\n')
    exec('def ylim(self,*args):\n  self.other_cmd(*args)\n')
    exec('def xlabel(self,*args):\n  self.other_cmd(*args)\n')
    exec('def ylabel(self,*args):\n  self.other_cmd(*args)\n')
    exec('def drawnow(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2d(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2d_init(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v3d_init(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2L(self,*args):\n  self.other_cmd(*args)\n')
    exec('def title(self,*args):\n  self.other_cmd(*args)\n')
    exec('def plot3(self,*args):\n  self.other_cmd(*args)\n')
    exec('def grid(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2dx(self,*args,**kargs):\n  self.other_cmd(*args,**kargs)\n')
    exec('def v3dx(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2d_show(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2d_line_object(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2d_clear(self,*args):\n  self.other_cmd(*args)\n')
    exec('def v2d_add_terrain(self,*args):\n  self.other_cmd(*args)\n')

    def __del__(self):
        # on the end of the program
        if hasattr(self, 'current_file_handle') and self.current_file_handle is not None:
            end_file_flag = (b'><EndTaskFlag\n')
            self.current_file_handle.write(end_file_flag)
            self.current_file_handle.close()
        if hasattr(self, 'port') and self.port is not None:
            self.disconnect()
        print蓝('the program exited, mcom as well exited!')

    def disconnect(self):
        self.socketx.close()



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
        msg = '@start@' + msg + '@end@'
        self.socketx.send(bytes(msg, encoding='utf8'))
        
    def __del__(self):
        self.socketx.close()






class MyHttp(Process):
    def __init__(self, path_to_html, avail_port):
        super(MyHttp, self).__init__()
        self.path_to_html = path_to_html
        self.avail_port = avail_port

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
        app.run(port=self.avail_port)
        





class DrawProcess(Process):
    def __init__(self, draw_udp_port, draw_mode):
        super(DrawProcess, self).__init__()
        self.draw_mode = draw_mode
        self.draw_udp_port = draw_udp_port
        self.draw_cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.draw_cmd_socket.bind(draw_udp_port)
        self.draw_cmd_socket.setblocking(False)
        return

    def init_matplot_lib(self):
        if self.draw_mode in ['Web', 'Img']:
            matplotlib.use('Agg') # set the backend before importing pyplot
            import matplotlib.pyplot as plt
            self.gui_reflesh = lambda: time.sleep(1) # plt.pause(0.1)
        elif self.draw_mode == 'Native':
            import matplotlib
            matplotlib.use('TkAgg') # set the backend before importing pyplot
            # matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            self.gui_reflesh = lambda: plt.pause(1)
        elif self.draw_mode == 'pyqtgraph':
            self.gui_reflesh = lambda: time.sleep(1) # plt.pause(0.1)
        else:
            assert False
        logdir = './TMP'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if self.draw_mode == 'Web':
            self.avail_port = find_free_port()
            my_http = MyHttp('%s/html.html'%logdir, self.avail_port)
            my_http.daemon = True
            my_http.start()
        self.libs_family = {
            'rec_init': 'rec', 'rec': 'rec', 'rec_show': 'rec',
            'v2d_init': 'v2d', 'v2dx':'v2d', 'v2d_show': 'v2d',
            'v2d_line_object':'v2d', 'v2d_clear':'v2d', 'v2d_add_terrain': 'v2d',
        }
        self.libs_init_fns = {
            'rec': self.rec_init_fn,
            'v2d': self.v2d_init_fn,
        }


    def run(self):
        self.init_matplot_lib()
        try:
            cnt = 0
            while True:
                try:
                    cmd_str, _ = self.draw_cmd_socket.recvfrom(4096)
                    self.process_cmd(cmd_str)
                    # print('成功处理指令:', cmd_str)
                except BlockingIOError:
                    self.gui_reflesh()
                    if self.draw_mode == 'Web':
                        cnt += 1
                        if cnt > 200: 
                            cnt = 0; print蓝('Web url: http://127.0.0.1:%d/'%self.avail_port)
                            

        except KeyboardInterrupt:
            self.__del__()
        self.__del__()

    def __del__(self):
        self.draw_cmd_socket.close()

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
            print蓝('绘图函数不能处理：', cmd)
            return None
        family_name = self.libs_family[func_name]
        if self.libs_init_fns[family_name] is not None:
            self.libs_init_fns[family_name]()
            self.libs_init_fns[family_name] = None
        return 'self.%s'%family_name

    # lib
    def rec_init_fn(self):
        from .mcom_rec import rec_family
        self.rec = rec_family('r', self.draw_mode)

    def v2d_init_fn(self):
        if self.draw_mode == 'pyqtgraph':
            from .mcom_v2d_new import v2d_family
        else:
            from .mcom_v2d import v2d_family
        self.v2d = v2d_family(self.draw_mode)

