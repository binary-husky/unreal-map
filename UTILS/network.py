import socket, threading, pickle, uuid, os

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
