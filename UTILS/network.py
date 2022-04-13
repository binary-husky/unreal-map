import socket, threading, pickle, uuid, os, atexit

def find_free_port():
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_host_ip():
    ip = None
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]
    finally:
        s.close()
    return ip


BUFSIZE = 10485760
# ip_port = ('127.0.0.1', 9999)
DEBUG_NETWORK = False
class UdpServer:
    def __init__(self, ip_port, obj='bytes') -> None:
        self.ip_port = ip_port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind(self.ip_port)
        self.most_recent_client = None
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        return

    def wait_next_dgram(self):
        data, self.most_recent_client = self.server.recvfrom(BUFSIZE)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('recv from :', self.most_recent_client, ' data :', data)
        return data

    def reply_last_client(self, data):
        assert self.most_recent_client is not None
        if DEBUG_NETWORK: print('reply_last_client :', self.most_recent_client, ' data :', data)
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.server.sendto(data, self.most_recent_client)
        return

    def __del__(self):
        self.server.close()
        return

class UdpTargetedClient:
    def __init__(self, target_ip_port, obj='bytes') -> None:
        self.target_ip_port = target_ip_port
        self.client = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        return

    def send_dgram_to_target(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.client.sendto(data, self.target_ip_port)
        if DEBUG_NETWORK: print('send_targeted_dgram :', self.target_ip_port, ' data :', data)
        return

    def send_and_wait_reply(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.client.sendto(data, self.target_ip_port)
        data, _ = self.client.recvfrom(BUFSIZE)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('get_reply :', self.target_ip_port, ' data :', data)
        return data


# ///////   test ipv4 udp
# import numpy as np

# server = UdpServer(ip_port, obj='pickle')
# client = UdpTargetedClient(ip_port, obj='pickle')

# def server_fn():
#     data = server.wait_next_dgram()
#     server.reply_last_client(np.array([4,5,6]))

# def client_fn():
#     rep = client.send_and_wait_reply(np.array([1,2,3]))


# thread_hi = threading.Thread(target=server_fn)
# thread_hello = threading.Thread(target=client_fn)
# # 启动线程
# thread_hi.start()
# thread_hello.start()

class UnixUdpServer:
    def __init__(self, unix_path, obj='bytes') -> None:
        try: os.makedirs(os.path.dirname(unix_path))
        except: pass
        self.unix_path = unix_path
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.server.bind(self.unix_path)
        self.most_recent_client = None
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        return

    def wait_next_dgram(self):
        data, self.most_recent_client = self.server.recvfrom(BUFSIZE)
        if DEBUG_NETWORK: print('self.most_recent_client',self.most_recent_client)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('recv from :', self.most_recent_client, ' data :', data)
        return data

    def reply_last_client(self, data):
        assert self.most_recent_client is not None
        if DEBUG_NETWORK: print('reply_last_client :', self.most_recent_client, ' data :', data)
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.server.sendto(data, self.most_recent_client)
        return

    def __del__(self):
        self.server.close()
        return

class UnixUdpTargetedClient:
    def __init__(self, target_unix_path, self_unix_path=None, obj='bytes') -> None:
        self.target_unix_path = target_unix_path
        if self_unix_path is not None:
            self.self_unix_path = self_unix_path  
        else:
            self.self_unix_path = target_unix_path+'_client_'+uuid.uuid1().hex[:5]
        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.client.bind(self.self_unix_path)
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        return

    def send_dgram_to_target(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.client.sendto(data, self.target_unix_path)
        if DEBUG_NETWORK: print('send_targeted_dgram :', self.target_unix_path, ' data :', data)
        return

    def send_and_wait_reply(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.client.sendto(data, self.target_unix_path)
        data, _ = self.client.recvfrom(BUFSIZE)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('get_reply :', self.target_unix_path, ' data :', data)
        return data


# ///////   test unix udp
# remote_uuid = uuid.uuid1().hex   # use uuid to identify threads

# unix_path = 'RECYCLE/Sockets/unix/%s'%remote_uuid
# server = UnixUdpServer(unix_path, obj='pickle')
# client = UnixUdpTargetedClient(unix_path, obj='pickle')

# def server_fn():
#     data = server.wait_next_dgram()
#     server.reply_last_client(np.array([4,5,6]))

# def client_fn():
#     rep = client.send_and_wait_reply(np.array([1,2,3]))


# thread_hi = threading.Thread(target=server_fn)
# thread_hello = threading.Thread(target=client_fn)
# # 启动线程
# thread_hi.start()
# thread_hello.start()



class StreamingPackageSep:
    def __init__(self):
        self.buff = [b'']
        self.myEOF = b'\xaa\x55\xaaHMP\xaa\x55'    # those bytes follow 010101 or 101010 pattern
        # self.myEOF = b'#A5@5A#'    # the EOF string for frame seperation

    def lower_send(self, data, connection):
        if DEBUG_NETWORK: assert self.myEOF not in data, 'This is (almost) not possible!'
        data = data + self.myEOF
        if DEBUG_NETWORK: print('data length:', len(data))
        connection.send(data)

    def lowest_recv(self, connection):
        while True:
            recvData = connection.recv(BUFSIZE)
            # ends_with_mark = recvData.endswith(self.myEOF)
            split_res = recvData.split(self.myEOF)
            assert len(split_res) != 0
            if len(split_res) == 1:
                # 说明没有终止符，直接将结果贴到buf最后一项
                self.buff[-1] = self.buff[-1] + split_res[0]
                if self.myEOF in self.buff[-1]: self.handle_flag_breakdown()
            else:
                n_split = len(split_res)
                for i, r in enumerate(split_res):
                    self.buff[-1] = self.buff[-1] + r   # 追加buff
                    if i == 0 and (self.myEOF in self.buff[-1]): 
                        # 第一次追加后，在修复的数据断面上发现了myEOF！
                        self.handle_flag_breakdown()
                    if i != n_split-1: 
                        # starts a new entry
                        self.buff.append(b'')
                    else:  
                        # i == n_split-1, which is the last item
                        if r == b'': continue
            if len(self.buff)>=2:
                # 数据成型，拿取成型的数据
                buff_list = self.buff[:-1]  
                self.buff = self.buff[-1:]
                return buff_list

    # Fox-Protocal
    def lower_recv(self, connection, expect_single=True):
        buff_list = self.lowest_recv(connection)
        if expect_single:
            assert len(buff_list) == 1, ('一次拿到了多帧数据, 但expect_single=True, 触发错误.')
            return buff_list[0], connection
        else:
            return buff_list, connection


    def handle_flag_breakdown(self):
        split_ = self.buff[-1].split(self.myEOF)
        assert len(split_)==2
        self.buff[-1] = split_[0]
        # starts a new entry
        self.buff.append(b'')
        self.buff[-1] = split_[1]
        return



# send() is used for TCP SOCK_STREAM connected sockets, and sendto() is used for UDP SOCK_DGRAM unconnected datagram sockets
class UnixTcpServerP2P(StreamingPackageSep):
    def __init__(self, unix_path, obj='bytes') -> None:
        super().__init__()
        try: os.makedirs(os.path.dirname(unix_path))
        except: pass
        self.unix_path = unix_path
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.unix_path)
        self.server.listen()
        self.most_recent_client = None
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        atexit.register(self.__del__)

    def accept_conn(self):
        conn, _  = self.server.accept()
        return conn

    def wait_next_dgram(self):
        if self.most_recent_client is None: self.most_recent_client, _ = self.server.accept()
        data, self.most_recent_client = self.lower_recv(self.most_recent_client)
        if DEBUG_NETWORK: print('self.most_recent_client',self.most_recent_client)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('recv from :', self.most_recent_client, ' data :', data)
        return data

    def reply_last_client(self, data):
        assert self.most_recent_client is not None
        if DEBUG_NETWORK: print('reply_last_client :', self.most_recent_client, ' data :', data)
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.lower_send(data, self.most_recent_client)
        return

    def __del__(self):
        self.server.close()
        try: os.remove(self.unix_path)
        except: pass
        return


class UnixTcpClientP2P(StreamingPackageSep):
    def __init__(self, target_unix_path, self_unix_path=None, obj='bytes') -> None:
        super().__init__()
        self.target_unix_path = target_unix_path
        if self_unix_path is not None:
            self.self_unix_path = self_unix_path  
        else:
            self.self_unix_path = target_unix_path+'_client_'+uuid.uuid1().hex[:5]
        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client.bind(self.self_unix_path)
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        self.connected = False
        atexit.register(self.__del__)

    def send_dgram_to_target(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        if not self.connected: self.client.connect(self.target_unix_path); self.connected = True
        self.lower_send(data, self.client)
        if DEBUG_NETWORK: print('send_targeted_dgram :', self.client, ' data :', data)
        return

    def send_and_wait_reply(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        if not self.connected: self.client.connect(self.target_unix_path); self.connected = True
        self.lower_send(data, self.client)
        data, _ = self.lower_recv(self.client)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('get_reply :', self.client, ' data :', data)
        return data

    def __del__(self):
        self.client.close()
        os.remove(self.self_unix_path)
        return

'''

    remote_uuid = uuid.uuid1().hex   # use uuid to identify threads

    unix_path = 'RECYCLE/Sockets/unix/%s'%remote_uuid
    server = UnixTcpServerP2P(unix_path, obj='pickle')
    client = UnixTcpClientP2P(unix_path, obj='pickle')

    def server_fn():
        # data = server.wait_next_dgram()
        # server.reply_last_client(np.array([4,5,6]))
        while 1:
            data = server.wait_next_dgram()
            server.reply_last_client(data)

    def client_fn():
        # rep = client.send_and_wait_reply(np.array([1,2,3]))
        while True:
            buf = np.random.rand(100,1000)
            rep = client.send_and_wait_reply(buf)
            assert (buf==rep).all()
            print('成功')


    thread_hi = threading.Thread(target=server_fn)
    thread_hello = threading.Thread(target=client_fn)
    # 启动线程
    thread_hi.start()
    thread_hello.start()

'''



# send() is used for TCP SOCK_STREAM connected sockets, and sendto() is used for UDP SOCK_DGRAM unconnected datagram sockets
class TcpServerP2P(StreamingPackageSep):
    def __init__(self, ip_port, obj='bytes') -> None:
        super().__init__()
        self.ip_port = ip_port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.ip_port)
        self.server.listen()
        self.most_recent_client = None
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        atexit.register(self.__del__)

    def accept_conn(self):
        conn, _  = self.server.accept()
        return conn

    def manual_wait_connection(self):
        if self.most_recent_client is None: 
            self.most_recent_client, _ = self.server.accept()
        return

    def wait_next_dgram(self):
        if self.most_recent_client is None: self.most_recent_client, _ = self.server.accept()
        data, self.most_recent_client = self.lower_recv(self.most_recent_client)
        if DEBUG_NETWORK: print('self.most_recent_client',self.most_recent_client)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('recv from :', self.most_recent_client, ' data :', data)
        return data

    def wait_multi_dgrams(self):
        if self.most_recent_client is None: self.most_recent_client, _ = self.server.accept()
        data_list, self.most_recent_client = self.lower_recv(self.most_recent_client, expect_single=False)
        if DEBUG_NETWORK: print('self.most_recent_client',self.most_recent_client)
        if self.convert_str: data_list = [data.decode('utf8') for data in data_list]
        if self.use_pickle: data_list = [pickle.loads(data) for data in data_list]
        if DEBUG_NETWORK: print('recv from :', self.most_recent_client, ' data_list :', data_list)
        return data_list

    def reply_last_client(self, data):
        assert self.most_recent_client is not None
        if DEBUG_NETWORK: print('reply_last_client :', self.most_recent_client, ' data :', data)
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        self.lower_send(data, self.most_recent_client)
        return

    def __del__(self):
        self.close()
        return

    def close(self):
        self.server.close()

class TcpClientP2P(StreamingPackageSep):
    def __init__(self, target_ip_port, self_ip_port=None, obj='bytes') -> None:
        super().__init__()
        self.target_ip_port = target_ip_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.use_pickle = (obj=='pickle')
        self.convert_str = (obj=='str')
        self.connected = False
        atexit.register(self.__del__)

    def send_dgram_to_target(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        if not self.connected: self.client.connect(self.target_ip_port); self.connected = True
        self.lower_send(data, self.client)
        if DEBUG_NETWORK: print('send_targeted_dgram :', self.client, ' data :', data)
        return

    def manual_connect(self):
        if not self.connected: self.client.connect(self.target_ip_port); self.connected = True

    def send_and_wait_reply(self, data):
        if self.use_pickle: data = pickle.dumps(data)
        if self.convert_str: data = bytes(data, encoding='utf8')
        if not self.connected: self.client.connect(self.target_ip_port); self.connected = True
        self.lower_send(data, self.client)
        data, _ = self.lower_recv(self.client)
        if self.convert_str: data = data.decode('utf8')
        if self.use_pickle: data = pickle.loads(data)
        if DEBUG_NETWORK: print('get_reply :', self.client, ' data :', data)
        return data

    def __del__(self):
        self.close()
        return

    def close(self):
        self.client.close()
'''

    ipport = ('127.0.0.1', 25453)
    server = TcpServerP2P(ipport, obj='pickle')
    client = TcpClientP2P(ipport, obj='pickle')
    def server_fn():
        data = server.wait_next_dgram()
        server.reply_last_client(np.array([4,5,6]))
    def client_fn():
        rep = client.send_and_wait_reply(np.array([1,2,3]))
    thread_hi = threading.Thread(target=server_fn)
    thread_hello = threading.Thread(target=client_fn)
    # 启动线程
    thread_hi.start()
    thread_hello.start()

'''



class QueueOnTcpClient():
    def __init__(self, ip):
        TCP_IP, TCP_PORT = ip.split(':')
        TCP_PORT = int(TCP_PORT)
        ip_port = (TCP_IP, TCP_PORT)

        self.tcpClientP2P = TcpClientP2P(ip_port, obj='str')
        self.tcpClientP2P.manual_connect()

    def send_str(self, b_msg):
        self.tcpClientP2P.send_dgram_to_target(b_msg)

    def close(self):
        self.tcpClientP2P.close()

    def __del__(self):
        self.close()

class QueueOnTcpServer():
    def __init__(self, ip_port):
        from UTILS.network import TcpServerP2P
        self.tcpServerP2P = TcpServerP2P(ip_port, obj='str')
        self.handler = None
        self.queue = None
        self.buff = ['']

    def wait_connection(self):
        self.tcpServerP2P.manual_wait_connection()
        t = threading.Thread(target=self.listening_thread)
        t.daemon = True
        t.start()

    def listening_thread(self):
        while True:
            buff_list = self.tcpServerP2P.wait_multi_dgrams()
            if self.handler is not None: 
                self.handler(buff_list)
            if self.queue is not None: 
                self.queue.put(buff_list)

    def set_handler(self, handler):
        self.handler = handler

    def get_queue(self):
        import queue
        self.queue = queue.Queue()
        return self.queue

    def recv(self):
        return

    def close(self):
        self.tcpServerP2P.close()

    def __del__(self):
        self.close()