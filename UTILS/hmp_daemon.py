import time, requests, threading, os, atexit
from UTILS.colorful import *
import psutil

def kill_process(p):
    try:
        # print('正在发送terminate命令到进程:', os.getpid(), '-->', p.pid)
        p.terminate()
        _, alive = psutil.wait_procs([p,], timeout=0.1)    # 先等 100ms
        if len(alive):
            _, alive = psutil.wait_procs(alive, timeout=3.0)  # 再等 3s
            if len(alive):
                # print('\t (R1) 很遗憾, 进程不服从terminate信号, 正在发送kill-9命令到进程:', os.getpid(), '-->', p.pid)
                for p in alive: p.kill()
            else:
                # print('\t (R2) 进程成功结束')
                pass
        else:
            # print('\t (R2) 进程成功结束')
            pass
    except Exception as e:
        print(e) 

def kill_process_and_its_children(p):
    p = psutil.Process(p.pid)   # p might be Python's process, convert to psutil's process
    if len(p.children())>0:
        # print('有子进程')
        for child in p.children():
            if hasattr(child,'children') and len(child.children())>0:
                kill_process_and_its_children(child)
            else:
                kill_process(child)
    else:
        pass
        # print('无子进程')
    kill_process(p)


def kill_process_children(p):
    p = psutil.Process(p.pid)   # p might be Python's process, convert to psutil's process
    if len(p.children())>0:
        # print('有子进程')
        for child in p.children():
            if hasattr(child,'children') and len(child.children())>0:
                kill_process_and_its_children(child)
            else:
                kill_process(child)
    else:
        pass
        # print('无子进程')

def clean_child_process(pid):
    parent = psutil.Process(pid)
    kill_process_children(parent)


def hmp_clean_up():
    from UTILS.exp_upload import upload_experiment_results
    from config import GlobalConfig as cfg
    print亮黄('[main.py] upload results to storage server via SSH')
    if cfg.allow_res_upload: upload_experiment_results(cfg)
    print亮黄('[main.py] kill all children process, then self-terminate.')

    clean_child_process(os.getpid())

def start_periodic_daemon(cfg):
    print('[hmp_daemon.py] Disable periodic daemon to debug.')
    return 
    periodic_thread = threading.Thread(target=periodic_daemon,args=(cfg,))

    periodic_thread.setDaemon(True)

    periodic_thread.start()
    for i in range(100):
        time.sleep(1)
        print(i)
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