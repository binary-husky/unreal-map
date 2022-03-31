from UTILS.exp_upload import get_ssh_sftp
from UTILS.colorful import *
import time,os

'''
Fetch experiment results from worker servers
'''

n_run_mode = [
    # {   # @1
    #     "addr": "172.18.116.149:2233",
    #     "usr": "hmp",
    #     "pwd": "hmp"
    # },
    # {   # @2
    #     "addr": "172.18.116.150:2233",
    #     "usr": "fuqingxu",
    #     "pwd": "clara"
    # },
    {   # @3
        "addr": "172.18.116.149:2233",
        "usr": "fuqingxu",
        "pwd": "clara"
    }

]
download_dir = './fetch/'
after_date = '2022-03-22-17-22-00'
consider_days = None

info_list = {}
to_download = {}
for ith_run in range(len(n_run_mode)):
    addr = n_run_mode[ith_run]['addr']
    usr = n_run_mode[ith_run]['usr']
    pwd = n_run_mode[ith_run]['pwd']
    ssh, sftp = get_ssh_sftp(addr, usr, pwd)
    experiments_path = sftp.listdir(path='./MultiServerMission/')
    # 将顺序改为从最早到最晚
    experiments_path = reversed(sorted(experiments_path))
    for index, exp_time in enumerate(experiments_path):
        time_then = time.mktime(time.strptime(exp_time,"%Y-%m-%d-%H:%M:%S"))
        time_now = time.mktime(time.localtime())
        diff_time_days = (time_now - time_then)/3600/24
        if consider_days is None:
            consider_days = (time_now - time.mktime(time.strptime(after_date,"%Y-%m-%d-%H-%M-%S")))/3600/24
        if diff_time_days > consider_days: continue
        path_ckpt = './MultiServerMission/%s/src/ZHECKPOINT/'%exp_time
        try:
            list_of_sub_exp = sftp.listdir(path=path_ckpt)
        except:
            print('No ZHECKPOINT directory found!')
            continue
        key = str(ith_run)+'-'+str(index)
        print亮绿(key,':',exp_time)
        for sep in list_of_sub_exp:
            print亮紫('\t- ',sep)
        info_list[key] = {'ith_run':ith_run, 'index':index, 'path':path_ckpt}
        target_path = (download_dir+'/%s/'%exp_time.replace(':','-'))
        try:
            os.mkdir(target_path)
            sftp.get_dir(source=path_ckpt,target=target_path)   # 下载！
        except BaseException as e:
            print('This directory already exists, skipping:', target_path)

print('download complete')
