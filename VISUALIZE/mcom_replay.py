# import os
# print(os.getcwd())
import os, sys
import argparse
from VISUALIZE.mcom import *



class RecallProcessThreejs(Process):
    def __init__(self, file_path):
        super(RecallProcessThreejs, self).__init__()
        self.buffer_list = []
        self.file_path = file_path

    def init_threejs(self):
        import threading
        t = threading.Thread(target=self.run_flask, args=(5051,))
        # t = threading.Thread(target=self.run_flask, args=(51241,))
        t.start()

    def run(self):
        self.init_threejs()
        try:
            with open(self.file_path, 'r') as f:
                new_buff_list = f.readlines()
            self.run_handler(new_buff_list)
            while True: time.sleep(1000)
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

        app = Flask(__name__)
        dirname = os.path.dirname(__file__) + '/threejsmod'
        import zlib

        @app.route("/up", methods=["POST"])
        def upvote():
            # dont send too much in one POST, might overload the network traffic
            if len(self.buffer_list)>35000:
                tosend = self.buffer_list[:30000]
                self.buffer_list = self.buffer_list[30000:]
            else:
                tosend = self.buffer_list
                self.buffer_list = []

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
    rp.start()
    rp.join()
