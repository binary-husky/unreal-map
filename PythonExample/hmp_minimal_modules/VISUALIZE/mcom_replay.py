# import os
# print(os.getcwd())
import os, sys, gzip
import argparse
from VISUALIZE.mcom import *

# DEBUG_OOM = True

class RecallProcessThreejs(Process):
    def __init__(self, file_path, port):
        super(RecallProcessThreejs, self).__init__()
        self.buffer_list = []
        self.file_path = file_path
        self.port = port
        self.client_send_pointer = {}

    def init_threejs(self):
        import threading
        t = threading.Thread(target=self.run_flask, args=(self.port,))
        t.daemon = True
        t.start()

    def __del__(self):
        pass
    
    def run(self):
        self.init_threejs()
        try:
            new_buff_list = []
            with gzip.open(self.file_path, 'rt') as zip:
                try:
                    for line in zip:
                        new_buff_list.append(line)
                        if len(new_buff_list) > 1e2:
                            self.run_handler(new_buff_list)
                            new_buff_list = []
                except:
                    print('File has bad ending! EOFError: Compressed file ended before the end-of-stream marker was reached!')
                    print('存档的末尾是破碎的, 少量数据可能丢失了. 完整的部分已经读取完成.')
            self.run_handler(new_buff_list)
            new_buff_list = []
            # if DEBUG_OOM:
            #     for i in range(100):
            #         print(i)
            #         with gzip.open(self.file_path, 'rt') as zip:
            #             try:
            #                 for line in zip:
            #                     if 'v2d_init' in line: continue
            #                     new_buff_list.append(line)
            #                     if len(new_buff_list) > 1e2:
            #                         self.run_handler(new_buff_list)
            #                         new_buff_list = []
            #             except:
            #                 print('File has bad ending! EOFError: Compressed file ended before the end-of-stream marker was reached!')
            #                 print('存档的末尾是破碎的, 少量数据可能丢失了. 完整的部分已经读取完成.')
            #         self.run_handler(new_buff_list)
            #         new_buff_list = []
            while True: 
                time.sleep(1000)
        except KeyboardInterrupt:
            self.__del__()

        self.__del__()

    def run_handler(self, new_buff_list):
        self.buffer_list.extend(new_buff_list)
        # too many, delete with fifo
        if len(self.buffer_list) > 1e9: # 当存储的指令超过十亿后，开始删除旧的
            del self.buffer_list[:len(new_buff_list)]

    def run_flask(self, port):
        from flask import Flask, url_for, jsonify, request, send_from_directory, redirect
        from waitress import serve
        from mimetypes import add_type
        add_type('application/javascript', '.js')
        add_type('text/css', '.css')

        app = Flask(__name__)
        dirname = os.path.dirname(__file__) + '/threejsmod'
        import zlib

        @app.route("/up", methods=["POST"])
        def upvote():
            # dont send too much in one POST, might overload the network traffic
            token = request.data.decode('utf8')
            if token not in self.client_send_pointer:
                print('[mcom_replay.py] Establishing new connection, token:', token)
                current_pointer = 0
            else:
                current_pointer = self.client_send_pointer[token]

            if len(self.buffer_list)-current_pointer>35000:
                tosend = self.buffer_list[current_pointer:current_pointer+30000]
                current_pointer = current_pointer+30000
            else:
                tosend = self.buffer_list[current_pointer:]
                current_pointer = len(self.buffer_list)
                
            self.client_send_pointer[token] = current_pointer

            # use zlib to compress output command, worked out like magic
            buf = "".join(tosend)
            buf = bytes(buf, encoding='utf8')   
            zlib_compress = zlib.compressobj()
            buf = zlib_compress.compress(buf) + zlib_compress.flush(zlib.Z_FINISH)
            return buf

        @app.route("/<path:path>")
        def static_dirx(path):
            if path=='favicon.ico': 
                return app.send_static_file('%s/files/favicon.ico'%dirname)
            return send_from_directory("%s/"%dirname, path)

        @app.route("/")
        def main_app():
            with open('%s/examples/abc.html'%dirname, 'r', encoding = "utf-8") as f:
                buf = f.read()
            return buf

        print('\n--------------------------------')
        print('JS visualizer online: http://%s:%d'%(get_host_ip(), port))
        print('JS visualizer online (localhost): http://localhost:%d'%(port))
        print('--------------------------------')
        # app.run(host='0.0.0.0', port=port)
        serve(app, threads=8, ipv4=True, ipv6=True, listen='*:%d'%port)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-p', '--path', help='directory of chosen file')
    args, unknown = parser.parse_known_args()
    if hasattr(args, 'path'):
        path = args.path
    else:
        assert False, (r"parser.add_argument('-p', '--path', help='The node name is?')")

    load_via_json = (hasattr(args, 'cfg') and args.cfg is not None)
    
    rp = RecallProcessThreejs(path)
    rp.daemon = True
    rp.start()
    rp.join()
