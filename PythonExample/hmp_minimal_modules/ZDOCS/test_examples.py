def validate_path():
    import os, sys
    dir_name = os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)
validate_path()

import glob
import subprocess
import copy, os
import time
import json
from UTIL.network import get_host_ip
from UTIL.colorful import *

test_subjects = glob.glob('ZDOCS/examples/**/*.jsonc',recursive=True)
print(test_subjects)

n_run = len(test_subjects)
target_server = [
    {
        "addr": "localhost:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run


def get_info(script_path):
    info = {
        'HostIP': get_host_ip(),
        'RunPath': os.getcwd(),
        'ScriptPath': os.path.abspath(script_path),
        'StartDateTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    try:
        info['DockerContainerHash'] = subprocess.getoutput(r'cat /proc/self/cgroup | grep -o -e "docker/.*"| head -n 1 |sed "s/docker\\/\\(.*\\)/\\1/" |cut -c1-12')
    except: 
        info['DockerContainerHash'] = 'None'
    return info


def run_batch_exp(n_run, n_run_mode, test_subjects, script_path, sum_note='run-hmp-test'):
    arg_base = ['python', 'main.py']
    time_mark_only = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    time_mark = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '-' + sum_note
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
        new_json_paths.append(test_subjects[i])

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

        print('byobu attach -t %s'%time_mark_only)
        addr_ip, addr_port = addr.split(':')
        print亮蓝("Attach cmd: ssh %s@%s -p %s -t \"byobu attach -t %s\""%(usr, addr_ip, addr_port, time_mark_only))
        
        stdin, stdout, stderr = ssh.exec_command(command='byobu new-session -d -s %s'%time_mark_only, timeout=1)
        print亮紫('byobu new-session -d -s %s'%time_mark_only)
        time.sleep(1)

        byobu_win_name = '%s--run-%d'%(time_mark_only, ith_run)
        byobu_win_name = byobu_win_name
        stdin, stdout, stderr = ssh.exec_command(command='byobu new-window -t %s'%time_mark_only, timeout=1)
        print亮紫('byobu new-window -t %s'%time_mark_only)
        time.sleep(1)

        cmd = 'cd  ' + src_path
        stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_only, cmd), timeout=1)
        print亮紫('byobu send-keys "%s" C-m'%cmd)
        time.sleep(1)

        
        cmd = ' '.join(['echo',  str(get_info(script_path)) ,'>>', './private_remote_execution.log'])
        stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_only, cmd), timeout=1)
        print亮紫('byobu send-keys "%s" C-m'%cmd)
        time.sleep(1)


        cmd = ' '.join(final_arg_list[ith_run])
        stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_only, cmd), timeout=1)
        print亮紫('byobu send-keys "%s" C-m'%cmd)
        time.sleep(1)

        print亮蓝("command send is done!")
        time.sleep(120)

        print亮蓝("kill")
        ssh.exec_command(command='byobu send-keys -t %s C-c'%(time_mark_only), timeout=1)
        time.sleep(2)
        ssh.exec_command(command='byobu send-keys -t %s C-c'%(time_mark_only), timeout=1)
        time.sleep(2)
        ssh.exec_command(command='byobu send-keys -t %s C-c'%(time_mark_only), timeout=1)
        print亮蓝("kill finish")

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

    
            
    input('Confirm execution? 确认执行?')
    input('Confirm execution! 确认执行!')

    t = 0
    while (t >= 0):
        print('Counting down ', t)
        time.sleep(1)
        t -= 1

    DELAY = 5
    for ith_run in range(n_run):
        worker(ith_run)
        for i in range(DELAY):
            print(f'\rrunning in {DELAY-i}', end='', flush=True)
            time.sleep(1)

    print('all submitted')


run_batch_exp(
    n_run=n_run, 
    n_run_mode=target_server, 
    test_subjects=test_subjects, 
    script_path=__file__, 
    sum_note='run-hmp-test')
