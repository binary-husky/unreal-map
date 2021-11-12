from flask import Flask, url_for, jsonify, request, send_from_directory, redirect
import json, glob
import numpy as np
import re, os
from multiprocessing import Process, Pipe



class FlaskProcess(Process):
    
    def __init__(self, port):
        super(FlaskProcess, self).__init__()
        self.port = port

    def run(self):
        app = Flask(__name__)
        dirname = os.path.dirname(__file__)

        @app.route("/up", methods=["POST"])
        def upvote():
            with open('%s/mcom.txt'%dirname, 'r', encoding = "utf-8") as f:
                buf = f.read()
            return buf

        @app.route("/<path:path>")
        def static_dirx(path):
            return send_from_directory("%s/"%dirname, path)

        @app.route("/")
        def main_app():
            with open('%s/examples/abc.html'%dirname, 'r', encoding = "utf-8") as f:
                buf = f.read()
            return buf

        app.run(host='0.0.0.0', port=self.port)