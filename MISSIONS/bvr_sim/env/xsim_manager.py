# -*- coding:UTF-8 -*-
"""
@FileName：xsim_manager.py
@Description：
@Author：liyelei
@Time：2021/5/25 9:34
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from UTILS.colorful import print亮红
import subprocess, atexit
import logging
import os, time
import re
import uuid

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

from UTILS.network import find_free_port

class XSimManager(object):
    def __init__(self, time_ratio: int, image_name: str = 'fuqingxu/bvrsim:trim'):
        self.xsim_time_ratio = time_ratio  # XSIM引擎加速比
        self.image_name = image_name  # 镜像名称
        # self.address = self.__isaddress(address)  # ip
        self.port = find_free_port()
        self.address = '127.0.0.1:'+str(self.port)
        # logging.info('当前引擎地址：{}'.format(self.address))
        self.domain_group = self.port
        self.xsim_run_num = self.port   # xsim环境运行编号(run_num)
        self.docker_name = 'xsim_' + str(self.xsim_run_num)  # 启动的容器名称
        self.solo_uuid = ''
        self.__start_env()
        atexit.register(self.__del__)

    def __del__(self):
        if hasattr(self,'_deleted_'): return    # avoid exit twice
        else: self._deleted_ = True     # avoid exit twice
        self.clean_container()


    def __start_env(self):
        '''
            -v, --volume list                    Bind mount a volume
            -w, --workdir string                 Working directory inside the container
            python daemon_server.py {self.xsim_time_ratio} {self.port} {HostID} {RTMNum}
        '''
        # solo_uuid = uuid.uuid1().hex
        # self.solo_uuid = solo_uuid
        RTMNum = 0 # 保存的回放次数，不使用原生回放系统
        # docker_run = "docker run --rm -p {}:{} -itd --name {} -w /home/x64 {} python daemon_server.py {} {} {}  {}"\
        #     .format(self.port, self.port, self.docker_name+solo_uuid[:5], self.image_name, self.xsim_time_ratio, self.port, self.domain_group, RTMNum)

        # 'docker run --rm --net host -p 40255:40255 -itd --name xsim_40255 -w /home/x64 fuqingxu/bvrsim:trim python daemon_server.py 100 40255 40255  0'
        # 'docker run --rm --net host -p 47191:47191 -itd --name xsim_47191 -w /home/x64 fuqingxu/bvrsim:trim python daemon_server.py 100 47191 47191  0'
        docker_run = "docker run --rm -p {}:{} -itd --name {} -w /home/x64 {} python daemon_server.py {} {} {} {}"\
            .format(self.port, self.port, self.docker_name, self.image_name, self.xsim_time_ratio, self.port, self.domain_group, RTMNum)
        print(docker_run)
        YOUR_ROOT_PASSWORD = 'hmp'
        os.system('echo %s|sudo -S %s' % (YOUR_ROOT_PASSWORD, docker_run))


    def clean_container(self):
        # import daemon
        # sudo docker stop -t 0 $(sudo docker ps | grep bvrsim | awk '{print $ 1}')
        YOUR_ROOT_PASSWORD = 'hmp'
        docker_stop = 'docker stop -t 0 ' + self.docker_name
        cmd = 'echo %s|sudo -S %s' % (YOUR_ROOT_PASSWORD, docker_stop)
        print亮红('[xsim_manager.py] os.system(%s)'%cmd)
        os.system(cmd)
        print亮红('[xsim_manager.py] 容器已经成功清除结束')





# def shut_down()
#     import subprocess
#     subprocess.Popen(open('RECYCLE/close_container'))

class AddressError(Exception):
    """IP地址无效异常"""
    pass


