import subprocess
import threading
import copy, os
import time
import json
from UTILS.colorful import *
# test sync to github
# ubuntu command to kill process: kill -9 $(ps -ef | grep xrdp | grep -v grep | awk '{print $ 2}')

arg_base = ['python', 'main.py']
log_dir = './T1-bench-IDL50/'
run_group = "bench"
# base_conf = 'train.json'

n_run = 3
conf_override = {
    "config.py->GlobalConfig-->note":       ["run-%d-%s"%(i+1, run_group)    for i in range(n_run)],
    "config.py->GlobalConfig-->seed":       [777+i                           for i in range(n_run)],
}

base_conf = {
    "config.py->GlobalConfig": {
        "env_name":"collective_assult",
        "env_path":"MISSIONS.collective_assult",
        "note": "benchIDL50",
        "num_threads": "50",
        "report_reward_interval": "50",
        "test_interval": "4096",
        "device": "cuda",
        "fold": "1",
        "seed": 777
    },

    "MISSIONS.collective_assult.collective_assult_parallel_run.py->ScenarioConfig": {
        "size": "5",
        "num_steps": "200",
        "render": "False",
        "TEAM_NAMES": [
            "ALGORITHM.hmp_ak_iagent.foundation->ReinforceAlgorithmFoundation"
        ]
    },

    "ALGORITHM.hmp_ak.foundation.py->AlgorithmConfig": {
        "train_traj_needed": "50"
    }
}

assert '_' not in run_group, ('下划线在matlab中的显示效果不好')
log_dir = log_dir+run_group
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

new_json_paths = []
for i in range(n_run):
    conf = copy.deepcopy(base_conf)
    new_json_path = '%s/run-%d.json'%(log_dir, i+1)
    for key in conf_override:
        tree_path, item = key.split('-->')
        conf[tree_path][item] = conf_override[key][i]
    with open(new_json_path,'w') as f:
        json.dump(conf, f, indent=4)
    print(conf)
    new_json_paths.append(new_json_path)











final_arg_list = []
printX = [print红,print绿,print黄,print蓝,print紫,print靛,print亮红,print亮绿,print亮黄,print亮蓝,print亮紫,print亮靛]

for ith_run in range(n_run):
    final_arg = copy.deepcopy(arg_base)
    final_arg.append('--cfg')
    final_arg.append(new_json_paths[ith_run])
    final_arg_list.append(final_arg)
    print('')

def worker(ith_run):
    log_path = open('PROFILE/%s/run-%d.log'%(log_dir, ith_run+1), 'w+')
    printX[ith_run%len(printX)](final_arg_list[ith_run])
    subprocess.run(final_arg_list[ith_run], stdout=log_path, stderr=log_path)

if __name__ == '__main__':
        
    input('确认执行？')
    input('确认执行！')

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
        for i in range(300):
            print('\r 错峰执行，启动倒计时%d     '%(300-i), end='', flush=True)
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
