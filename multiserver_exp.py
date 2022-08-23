import numpy as np
# ubuntu command to kill process: kill -9 $(ps -ef | grep python | grep hmp | grep -v grep | awk '{print $ 2}')

n_run = 3
n_run_mode = [
    {
        "addr": "exe_here=>localhost:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*3
assert len(n_run_mode)==n_run

conf_override = {
    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],
    "config.py->GlobalConfig-->note":
        [
            "RVE-drone1-fixaa-run1",
            "RVE-drone1-fixaa-run2",
            "RVE-drone1-fixaa-run3",
        ],

}


true = True
false = False

base_conf =  {
    #  --- Part1: config HMP core --- 
    "config.py->GlobalConfig": {
        "note": "z-obsbreak-RVE-drone=0",
        "env_name": "uhmap",
        "env_path": "MISSION.uhmap",
        "draw_mode": "Img",
        "num_threads": 16,
        "report_reward_interval": 256,
        "test_interval": 5120,
        "test_epoch": 512,
        "interested_team": 0,
        "seed": 10088,
        "device": "cuda",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "ALGORITHM/conc_4hist_hete",
            "MISSION/uhmap"
        ]
    },


    #  --- Part2: config MISSION --- 
    "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {
        "n_team1agent": 10,
        "n_team2agent": 10,
        "MaxEpisodeStep": 125,
        "StepGameTime": 0.5,
        "StateProvided": false,
        "render": false, #  note: random seed has different impact on renderer and server
        "UElink2editor": false,
        "AutoPortOverride": true,
        "HeteAgents": true,
        "UnrealLevel": "UhmapLargeScale",
        "SubTaskSelection": "UhmapLargeScale",
        #  "UhmapRenderExe": "./../../WindowsNoEditor/UHMP.exe",
        #  "UhmapServerExe": "./../../WindowsServer/UHMPServer.exe",
        "UhmapRenderExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxNoEditor/UHMP.sh",
        "UhmapServerExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxServer/UHMPServer.sh",
        "TimeDilation": 64, #  simulation time speed up, larger is faster
        "TEAM_NAMES": [
            "ALGORITHM.conc_4hist_hete.foundation->ReinforceAlgorithmFoundation",
            "ALGORITHM.script_ai.uhmap_ls->DummyAlgorithmLinedAttack",
        ]
    },
    "MISSION.uhmap.SubTasks.UhmapLargeScaleConf.py->SubTaskConfig":{
        "agent_list": [
            { "team":0,  "tid":0,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },

            { "team":1,  "tid":0,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
        ]
    },






    #  --- Part3: config ALGORITHM 1/2 --- 
    "ALGORITHM.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },

    #  --- Part3: config ALGORITHM 2/2 --- 
    "ALGORITHM.conc_4hist_hete.shell_env.py->ShellEnvConfig": {
        "add_avail_act": true
    },
    "ALGORITHM.conc_4hist_hete.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 256,
        "hete_type_trainable": [true, true, true],
        "load_checkpoint": false,
        "load_specific_checkpoint": "",
        "n_focus_on": 3,
        "gamma": 0.99,
        "gamma_in_reward_forwarding": "True",
        "gamma_in_reward_forwarding_value": 0.95,
        "prevent_batchsize_oom": "True",
        "lr": 0.0004,
        "ppo_epoch": 24,
        "policy_resonance": "False",
        "n_entity_placeholder": 22
    },
    # "ALGORITHM.conc_4hist_hete.stage_planner.py->PolicyRsnConfig": {
    #     "resonance_start_at_update": None,
    #     "yita_min_prob": None,
    #     "yita_max": None,
    #     "yita_inc_per_update": None,
    # }

    
}
##############################################################################
##############################################################################
##############################################################################

import subprocess
import threading
import copy, os
import time
import json
from UTIL.colorful import *
arg_base = ['python', 'main.py']
time_mark = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
log_dir = '%s/'%time_mark
exp_log_dir = log_dir+'exp_log'
if not os.path.exists('PROFILE/%s'%exp_log_dir):
    os.makedirs('PROFILE/%s'%exp_log_dir)
exp_json_dir = log_dir+'exp_json'
if not os.path.exists('PROFILE/%s'%exp_json_dir):
    os.makedirs('PROFILE/%s'%exp_json_dir)

conf_list = []
new_json_paths = []
for i in range(n_run):
    conf = copy.deepcopy(base_conf)
    new_json_path = 'PROFILE/%s/run-%d.json'%(exp_json_dir, i+1)
    for key in conf_override:
        assert n_run == len(conf_override[key]), ('检查！n_run是否对应')
        tree_path, item = key.split('-->')
        conf[tree_path][item] = conf_override[key][i]
    with open(new_json_path,'w') as f:
        json.dump(conf, f, indent=4)
    # print(conf)
    conf_list.append(conf)
    new_json_paths.append(new_json_path)

print红('\n')
print红('\n')
print红('\n')

printX = [
    print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
    print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
    print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
    print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
    print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
    print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
    print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
    print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
    print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
    print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
]
conf_base_ = conf_list[0]
for k_ in conf_base_:
    conf_base = conf_base_[k_]
    for key in conf_base:
        different = False
        for i in range(len(conf_list)):
            if conf_base[key]!=conf_list[i][k_][key]:
                different = True
                break
        # 
        if different:
            for i in range(len(conf_list)):
                printX[i](key, conf_list[i][k_][key])
        else:
            print(key, conf_base[key])



final_arg_list = []

for ith_run in range(n_run):
    final_arg = copy.deepcopy(arg_base)
    final_arg.append('--cfg')
    final_arg.append(new_json_paths[ith_run])
    final_arg_list.append(final_arg)
    print('')


def local_worker(ith_run):
    log_path = open('PROFILE/%s/run-%d.log'%(exp_log_dir, ith_run+1), 'w+')
    printX[ith_run%len(printX)](final_arg_list[ith_run])
    subprocess.run(final_arg_list[ith_run], stdout=log_path, stderr=log_path)

def remote_worker(ith_run):
    # step 1: transfer all files
    from UTIL.exp_helper import get_ssh_sftp
    
    addr = n_run_mode[ith_run]['addr']
    if 'exe_here' in addr: 
        _, addr = addr.split('=>')
        usr = n_run_mode[ith_run]['usr']
        pwd = n_run_mode[ith_run]['pwd']
        ssh, sftp = get_ssh_sftp(addr, usr, pwd)
        src_path = os.getcwd()
    else:
        # assert False
        usr = n_run_mode[ith_run]['usr']
        pwd = n_run_mode[ith_run]['pwd']
        ssh, sftp = get_ssh_sftp(addr, usr, pwd)
        sftp.mkdir('/home/%s/MultiServerMission'%(usr), ignore_existing=True)
        sftp.mkdir('/home/%s/MultiServerMission/%s'%(usr, time_mark), ignore_existing=True)
        src_path = '/home/%s/MultiServerMission/%s/src'%(usr, time_mark)
        try:
            sftp.mkdir(src_path, ignore_existing=False)
            sftp.put_dir('./', src_path, ignore_list=['.vscode', '__pycache__','TEMP','ZHECKPOINT'])
            sftp.close()
            print紫('upload complete')
        except:
            sftp.close()
            print紫('do not need upload')

    time_mark_ = time_mark.replace(':','-')
    print('byobu attach -t %s'%time_mark_)

    stdin, stdout, stderr = ssh.exec_command(command='byobu new-session -d -s %s'%time_mark_, timeout=1)
    print亮紫('byobu new-session -d -s %s'%time_mark_)
    time.sleep(1)

    byobu_win_name = '%s--run-%d'%(time_mark_, ith_run)
    byobu_win_name = byobu_win_name.replace(':','-')
    stdin, stdout, stderr = ssh.exec_command(command='byobu new-window -t %s'%time_mark_, timeout=1)
    print亮紫('byobu new-window -t %s'%time_mark_)
    time.sleep(1)


    cmd = 'cd  ' + src_path
    stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_, cmd), timeout=1)
    print亮紫('byobu send-keys "%s" C-m'%cmd)
    time.sleep(1)

    cmd = ' '.join(final_arg_list[ith_run])
    stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_, cmd), timeout=1)
    print亮紫('byobu send-keys "%s" C-m'%cmd)
    time.sleep(1)

    # 杀死
    # stdin, stdout, stderr = ssh.exec_command(command='byobu kill-session -t %s'%byobu_win_name, timeout=1)
    pass

def worker(ith_run):
    if n_run_mode[ith_run] is None: 
        local_worker(ith_run)
    else:
        remote_worker(ith_run)



def clean_process(pid):
    import psutil
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        try:
            print亮红('sending Terminate signal to', child)
            child.terminate()
            time.sleep(5)
            print亮红('sending Kill signal to', child)
            child.kill()
        except: pass
    parent.kill()

def clean_up():
    print亮红('clean up!')
    parent_pid = os.getpid()   # my example
    clean_process(parent_pid)

DELAY = 10

if __name__ == '__main__':
        
    input('Confirm execution? 确认执行?')
    input('Confirm execution! 确认执行!')

    t = 0
    while (t >= 0):
        print('Counting down ', t)
        time.sleep(1)
        t -= 1

    for ith_run in range(n_run):
        worker(ith_run)
        for i in range(DELAY):
            time.sleep(1)

    print('all submitted')