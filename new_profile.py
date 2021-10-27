import subprocess
import threading
import copy, os
import time
import json
import glob
from UTILS.colorful import *
import datetime
# ubuntu command to kill process: kill -9 $(ps -ef | grep xrdp | grep -v grep | awk '{print $ 2}')
# def read_json_withcomment(fp):
#     from jsoncomment import JsonComment
#     json_creader = JsonComment()
#     json_obj = json_creader.load(fp)
#     return json_obj
json_path = './RUN/*.json'








arg_base = ['python', 'main.py']
path_all = glob.glob(json_path)
run_profile = []
ignore_key = ['gpu_party','device']
lines_previous = None
for path in path_all:

    with open(path, 'r') as f:
        config = json.loads(f.read())
    with open(path, 'r') as f:
        lines = f.readlines()
        
    note = config["config.py->GlobalConfig"]['note']
    print('--------------------')
    print亮紫('inspecting:', note)
    if lines_previous is not None:
        for L in lines:
            if L not in lines_previous:
                if not any([k in L for k in ignore_key]):
                    print紫(L.replace('\n',''))
    else:
        print紫('reading first config')
    lines_previous = lines

    t_profile = {
        'json_path': path,
        'note': note,
        'conf': config
    }
    run_profile.append(t_profile)
    input('confirm? manual run with: python main.py --cfg %s'%path)

n_run = len(run_profile)
final_arg_list = []
printX = [  print红,print绿,print黄,print蓝,print紫,print靛,print亮红,
            print亮绿,print亮黄,print亮蓝,print亮紫,print亮靛]

log_dir = './PROFILE/%s'%(datetime.datetime.now())
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def worker(ith_run):
    profile = run_profile[ith_run]
    note = profile['note']
    log_path = open('%s/%s.log'%(log_dir, note), 'w+')
    final_arg_list = ['python', 'main.py', '--cfg', profile['json_path']]
    printX[ith_run%len(printX)](final_arg_list)
    subprocess.run(final_arg_list, stdout=log_path, stderr=log_path)

def profile():
    input('确认执行？')
    t = 0
    while (t >= 0):
        print('运行倒计时：', t)
        time.sleep(1)
        t -= 1
    threads = [ threading.Thread( target=worker,args=(ith_run,) ) for ith_run in range(n_run) ]
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
        print('错峰执行，启动', thread)
        for i in range(3):
            print('\r 错峰执行，启动倒计时%d     '%(3-i), end='', flush=True)
            time.sleep(1)

    while True:
        is_alive = [thread.is_alive() for thread in threads]
        if any(is_alive):
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            print(time_now, 'I am still running!', is_alive)
            print靛('current scipt:%s, current log:%s'%(os.path.abspath(__file__), log_dir))
            time.sleep(120)
        else:
            break
    print('[profile] All task done!')
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    profile()