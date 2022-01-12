import time, requests, threading, os, atexit
from UTILS.colorful import *

def clean_child_process(pid):
    import psutil, time
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        try:
            print('sending Terminate signal to', child)
            child.terminate()
            time.sleep(1)
            print('sending Kill signal to', child)
            child.kill()
        except: pass
    parent.kill()

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