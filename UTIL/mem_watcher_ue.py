def validate_path():
    import os, sys
    # '/home/hmp/xx/hmp2g-heterogeneous-phase2/UTIL'
    dir_name = os.path.dirname(__file__)
    # '/home/hmp/xx/hmp2g-heterogeneous-phase2'
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    # change working dir
    os.chdir(root_dir_assume)
    # import root
    sys.path.append(root_dir_assume)
    
validate_path()
import time, requests, threading, os, atexit, psutil
from UTIL.colorful import *

def thread_dfs(p, depth=0, info=None):
    try:
        if isinstance(p, int):
            p = psutil.Process(p)
        elif isinstance(p, psutil.Process):
            pass
        else:
            p = psutil.Process(p.pid)
        
        pp = p
        print_info(depth, pp, info)
        if len(p.children())>0:
            # print('有子进程')
            for child in p.children():
                if hasattr(child,'children') and len(child.children())>0:
                    thread_dfs(child, depth = depth+1, info=info)
                else:
                    pp = child
                    print_info(depth+1, pp, info)
        else:
            pass
    except:
        return

def print_info(depth, pp, info=None):
    pid = pp.pid
    name = pp.name()
    name_trim = 'HmapShmPoolWorker'  if name.startswith('HmapShmPoolWorker') else name
    mem = (psutil.Process(pid).memory_info().rss / 1024 / 1024 / 1024)
    info['tot_mem'] += mem
    info['tot_procs'] += 1
    if name_trim not in info:
        info[name_trim] = {
            'mem':0,
            'procs':0,
        }
    info[name_trim]['mem'] += mem
    info[name_trim]['procs'] += 1


def find_procs_by_name(name):
    "Return a list of processes matching 'name'."
    ls = []
    for p in psutil.process_iter(["name", "exe", "cmdline"]):
        if name == p.info['name'] or \
                p.info['exe'] and os.path.basename(p.info['exe']) == name or \
                p.info['cmdline'] and p.info['cmdline'][0] == name:
            ls.append(p)
    return ls[0]

if __name__ ==  "__main__":
    from VISUALIZE.mcom import mcom
    mcv = mcom( 
            path='TEMP',
            digit=-1, 
            rapid_flush=True, draw_mode='Img'
    )


def main(root_name = 'UE4Editor.exe'):
    pid = find_procs_by_name(root_name)
    mem = (pid.memory_info().rss / 1024 / 1024 / 1024)
    mcv.rec(mem, 'mem')
    mcv.rec_show()

if __name__ ==  "__main__":
    while True:
        main()
        time.sleep(10) # 十分钟一次
        # time.sleep(300) # 十分钟一次