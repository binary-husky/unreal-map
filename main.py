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
"""
def SET_NUM_THREADS(internal_threads):
    import os
    os.environ['NUM_THREADS'] = str(internal_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(internal_threads)
    os.environ['MKL_NUM_THREADS'] = str(internal_threads)
    os.environ['OMP_NUM_THREADS'] = str(internal_threads)
SET_NUM_THREADS(1)

def pytorch_gpu_init(cfg, internal_threads):
    import os, torch
    from UTILS.auto_gpu import sel_gpu
    torch.set_num_threads(int(internal_threads))
    seed = cfg.seed; device = cfg.device
    torch.manual_seed(seed)
    # e.g. device='cpu
    if not 'cuda' in device: return
    # removing this assert can cause error?
    assert torch.cuda.is_available()
    if 'm-cuda' in device: assert False # m-gpu is not functional yet
    if device == 'cuda': gpu_index = sel_gpu().auto_choice()
    else: # e.g. device='cuda:0'
        gpu_index = int(device.split(':')[-1])
        cfg.manual_gpu_ctl = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    torch.cuda.set_device(gpu_index)
    cfg.device = 'cuda:%d'%gpu_index
    torch.cuda.manual_seed(seed)



if __name__ == '__main__':
    import os, numpy
    import pyximport; pyximport.install(build_dir='./RECYCLE/build/', inplace=True, language_level=3)
    from atexit import register
    from UTILS.colorful import *
    from UTILS.config_args import get_args
    from UTILS.shm_pool import SmartPool
    cfg = get_args()
    
    # Set numpy seed
    numpy.random.seed(cfg.seed)
    numpy.set_printoptions(3, suppress=True)

    # Get mem-sharing process pool
    assert cfg.num_threads % cfg.fold == 0, ('Use n process to run n*m parallel threads!')
    smart_pool = SmartPool(fold=cfg.fold, proc_num=cfg.num_threads // cfg.fold, base_seed=cfg.seed)
    register(smart_pool.party_over)  # Failsafe, handles shm leak

    # Pytorch has to be init AFTER the process pool starts, set pytorch seed
    pytorch_gpu_init(cfg=cfg, internal_threads=os.environ['NUM_THREADS'])

    # Prepare everything else
    from task_runner import Runner
    runner = Runner(process_pool=smart_pool)

    # GO! GO! GO!
    runner.run()

    # DONE!
    smart_pool.party_over()
    print绿('All jobs finished')

