# This Python file uses the following encoding: utf-8
"""
    Author: Fu Qingxu,CASIA
    Description: Enterance for everything in HMP
    In file you can find:
        1.Config-Parsing; 2.Multiprocess-Init
        3.GPU-Selection;  4.Seed-Setting
    If you are interested in something, you may continue to read:
        Handling parallel environment             -->   task_runner.py & shm_env.py
        Link between teams and diverse algorithms -->   multi_team.py
        Adding new env                            -->   MISSIONS.env_router.py
        Adding algorithm                          -->   ALGORITHM.example_foundation.py
        Configuring by writing py files           -->   config.py
        Configuring by json                       -->   xx.json
        colorful printing                         -->   colorful.py
        auto pip deployer                         -->   pip_find_missing.py
        efficient parallel execting               -->   shm_pool.pyx
        auto gpu selection                        -->   auto_gpu.py
        matlab logging/plotting bridge            -->   mcom.py & mcom_rec.py
        experiment batch executor                 -->   mprofile.py
        1->3,  2->5, 3->7
"""
import os, sys
def SET_NUM_THREADS(internal_threads):
    os.environ['NUM_THREADS'] = str(internal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(internal_threads)
    os.environ['MKL_NUM_THREADS'] = str(internal_threads)
    os.environ['OMP_NUM_THREADS'] = str(internal_threads)
SET_NUM_THREADS(1)

# do NOT edit this func
def pytorch_gpu_init(cfg):
    import torch
    from UTILS.auto_gpu import sel_gpu
    torch.set_num_threads(int(os.environ['NUM_THREADS']))
    seed = cfg.seed; device = cfg.device
    torch.manual_seed(seed)
    # e.g. device='cpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if not 'cuda' in device: return
    if 'm-cuda' in device: assert False # m-gpu is not functional yet
    if device == 'cuda': gpu_index = sel_gpu().auto_choice()
    else: # e.g. device='cuda:0'
        gpu_index = int(device.split(':')[-1])
        cfg.manual_gpu_ctl = True
        if cfg.gpu_fraction!=1: torch.cuda.set_per_process_memory_fraction(cfg.gpu_fraction, gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    cfg.device = 'cuda' # remove ':x', the selected gpu is cuda:0 from now on
    torch.cuda.manual_seed(seed)
    if cfg.use_float64:
        torch.set_default_dtype(torch.float64)

def clean_up():
    print('upload results to storage server via SSH')
    from UTILS.exp_upload import upload_experiment_results
    from UTILS.shm_pool import clean_child_process
    if cfg.allow_res_upload: upload_experiment_results(cfg)
    print('kill all children process')
    clean_child_process(os.getpid())

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


if __name__ == '__main__':
    # import tracemalloc; tracemalloc.start(); import linecache
    
    import numpy
    import pyximport; pyximport.install(build_dir='./RECYCLE/build/', inplace=True, language_level=3, setup_args={'include_dirs': numpy.get_include()})
    from atexit import register
    from UTILS.colorful import *
    from UTILS.config_args import prepare_args
    from UTILS.shm_pool import SmartPool
    cfg = prepare_args()
    register(clean_up)  # Failsafe, handles mem leak

    # Set numpy seed
    numpy.random.seed(cfg.seed)
    numpy.set_printoptions(3, suppress=True)

    # Get mem-sharing process pool
    assert cfg.num_threads % cfg.fold == 0, ('Use n process to run n*m parallel threads!')
    smart_pool = SmartPool(fold=cfg.fold, proc_num=cfg.num_threads // cfg.fold, base_seed=cfg.seed)
    register(smart_pool.party_over)  # exe first Failsafe, handles shm leak

    # Pytorch has to be init AFTER the process pool starts, set pytorch seed
    pytorch_gpu_init(cfg=cfg)
    # 

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)

    # Prepare everything else
    from task_runner import Runner
    # import time; time.sleep(999)
    runner = Runner(process_pool=smart_pool)
    # GO! GO! GO!
    runner.run() 

    # DONE!
    print绿('--- All jobs finished ---')
    smart_pool.party_over()
