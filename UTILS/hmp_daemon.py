import time, requests, threading, os, atexit
from UTILS.colorful import *


def kill_process(p):
    try:
        print('正在发送terminate命令到进程:', p.pid)
        p.terminate()
        print('正在发送kill命令到进程:', p.pid)
        time.sleep(3)
        p.kill()
    except: pass

def kill_process_and_its_children(p):
    print('开始杀死以下进程及其子进程:',p.pid)
    for child in p.children():
        print('->', child.pid)

    for child in p.children():
        if hasattr(child,'children') and len(child.children())>0:
            kill_process_and_its_children(child)
        else:
            kill_process(child)
    kill_process(p)



def clean_child_process(pid):
    import psutil, time
    parent = psutil.Process(pid)
    kill_process_and_its_children(parent)


def hmp_clean_up():
    from UTILS.exp_upload import upload_experiment_results
    from config import GlobalConfig as cfg
    print亮黄('[main.py] upload results to storage server via SSH')
    if cfg.allow_res_upload: upload_experiment_results(cfg)
    print亮黄('[main.py] kill all children process, then self-terminate.')
    clean_child_process(os.getpid())

def start_periodic_daemon(cfg):
    periodic_thread = threading.Thread(target=periodic_daemon,args=(cfg,))
    periodic_thread.setDaemon(True)
    periodic_thread.start()
    atexit.register(hmp_clean_up)

def periodic_daemon(cfg):
    while True:
        try: 
            print('start periodic_daemon_(cfg)')
            periodic_daemon_(cfg)
            print('end periodic_daemon_(cfg)')
        except AssertionError: 
            hmp_clean_up()
        except BaseException: 
            print('hmp server failed')
            break
        time.sleep(15*60)

def periodic_daemon_(cfg):
    report = {
        'type': 'hmp-client',
        'note': cfg.note,
        'time': time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
        'client_status': 'Running',
        'StartingTime': cfg.machine_info['StartDateTime'],
        'HostIP': cfg.machine_info['HostIP'],
        'ExpUUID': cfg.machine_info['ExpUUID'],
        'RunPath':cfg.machine_info['RunPath'],
        'DockerContainerHash':cfg.machine_info['DockerContainerHash']
    }
    res = requests.post('http://linux.ipv4.fuqingxu.top:11511/',data = report)

    if res.text=='Stop_Now': 
        report['client_status'] = 'Terminate'
        requests.post('http://linux.ipv4.fuqingxu.top:11511/',data = report)
        raise AssertionError('HMP-Center Has Given Terminate Signal!')