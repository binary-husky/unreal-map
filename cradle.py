import threading, uuid
import numpy as np
from UTIL.network import UnixUdpServer, UnixUdpTargetedClient

remote_uuid = uuid.uuid1().hex   # use uuid to identify threads

unix_path = 'TEMP/Sockets/unix/%s'%remote_uuid
server = UnixUdpServer(unix_path, obj='pickle')
client = UnixUdpTargetedClient(unix_path, obj='pickle')

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