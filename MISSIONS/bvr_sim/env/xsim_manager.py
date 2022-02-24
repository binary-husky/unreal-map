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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

from UTILS.network import find_free_port
class XSimManager(object):
    def __init__(self, time_ratio: int, image_name: str = 'fuqingxu/bvrsim:latest'):
        self.xsim_time_ratio = time_ratio  # XSIM引擎加速比
        self.image_name = image_name  # 镜像名称
        # self.address = self.__isaddress(address)  # ip
        self.address = '127.0.0.1'
        self.port = find_free_port()
        # logging.info('当前引擎地址：{}'.format(self.address))
        self.domain_group = self.port
        self.xsim_run_num = self.port   # xsim环境运行编号(run_num)
        self.docker_name = 'xsim_' + str(self.xsim_run_num)  # 启动的容器名称
        self.solo_uuid = ''
        self.__start_env()

    def __del__(self):
        print亮红('调用XSimManager的__del__')
        if hasattr(self,'_deleted_'): return    # avoid exit twice
        else: self._deleted_ = True     # avoid exit twice
        self.close_env()


    def __start_env(self):

        '''
            -v, --volume list                    Bind mount a volume
            -w, --workdir string                 Working directory inside the container
            python daemon_server.py {self.xsim_time_ratio} {self.port} {HostID} {RTMNum}
        '''
        solo_uuid = uuid.uuid1().hex
        self.solo_uuid = solo_uuid
        RTMNum = 0 # 保存的回放次数，不使用原生回放系统
        docker_run = "docker run --rm -p {}:{} -itd --name {} -w /home/x64 {} python daemon_server.py {} {} {}  {}"\
            .format(self.port, self.port, self.docker_name+solo_uuid[:5], self.image_name, self.xsim_time_ratio, self.port, self.domain_group, RTMNum)

        YOUR_ROOT_PASSWORD = 'hmp'
        os.system('echo %s|sudo -S %s' % (YOUR_ROOT_PASSWORD, docker_run))

        atexit.register(self.clean_container)

    def clean_container(self):
        # import daemon
        # sudo docker stop -t 0 $(sudo docker ps | grep bvrsim | awk '{print $ 1}')
        YOUR_ROOT_PASSWORD = 'hmp'
        docker_stop = 'docker stop -t 0 ' + self.docker_name + self.solo_uuid[:5]
        cmd = 'echo %s|sudo -S %s' % (YOUR_ROOT_PASSWORD, docker_stop)
        print亮红('os.system(cmd) 开始')
        os.system(cmd)
        print亮红('os.system(cmd) 结束')



    def close_env(self):
        self.clean_container()


# def shut_down()
#     import subprocess
#     subprocess.Popen(open('RECYCLE/close_container'))

class AddressError(Exception):
    """IP地址无效异常"""
    pass