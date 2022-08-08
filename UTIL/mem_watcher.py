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
    return ls

from VISUALIZE.mcom import mcom
mcv = mcom( 
        path='TEMP',
        digit=-1, 
        rapid_flush=True, draw_mode='Img'
)


def main(root_name = 'HmapRootProcess'):
    res = find_procs_by_name(root_name)
    info_list = []
    for r in res:
        info = {
            'tot_mem':0,
            'tot_procs':0,
        }
        thread_dfs(r, 0, info)
        info_list.append(info)
        
    if len(info_list)>0:
        print(info_list)
        info = info_list[0]
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(info)
        mcv.rec(info['tot_mem'], 'mem')
        mcv.rec(info['HmapRootProcess']['mem'], 'HmapRootProcess')
        mcv.rec(info['HmapShmPoolWorker']['mem'], 'HmapShmPoolWorker')
        mcv.rec(info['UHMPServer-Linux-Test']['mem'], 'UHMPServer-Linux-Test')
        mcv.rec(info['UHMPServer.sh']['mem'], 'UHMPServer.sh')
        mcv.rec(info['python3']['mem'], 'python3')
        mcv.rec_show()

while True:
    main()
    time.sleep(10)