import argparse, os, time, func_timeout
from shutil import copyfile, copytree, ignore_patterns
from .colorful import *

'''
    This a chained var class, it deal with hyper-parameters that are bound together, 
    e.g. number of threads and test episode interval.
    ChainVars are handled in utils.config_args.py
'''
class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with

def secure_chained_vars(default_cfg, new_cfg, vb):
    default_cfg_dict = default_cfg.__dict__
    altered_cv = []
    for key in default_cfg_dict:
        if not key.endswith('_cv'): continue
        o_key = key.replace('_cv','')
        if o_key in new_cfg: continue
        assert hasattr(default_cfg, o_key), ('twin var does not have original')
        # get twin
        chain_var = getattr(default_cfg, key)   
        need_reflesh = False
        for chain_by_var in chain_var.chained_with:
            if chain_by_var in new_cfg: need_reflesh = True
        if not need_reflesh: continue
        replace_item = chain_var.chain_func(*[getattr(default_cfg, v) for v in chain_var.chained_with])
        original_item = getattr(default_cfg, o_key)
        if vb: print靛('[config] warning, %s is chained by %s, automatic modifying:'%(o_key,
                            str(chain_var.chained_with)), original_item, '-->', replace_item)
        setattr(default_cfg, o_key, replace_item)
        altered_cv.append(o_key)
    return altered_cv

def override_config_file(cfg_group, new_cfg, vb):
    import importlib
    assert '->' in cfg_group
    str_pro = '------------- %s -------------'%cfg_group
    if vb:  print绿(str_pro)
    file_, class_ = cfg_group.split('->')
    if '.py' in file_: 
        # replace it with removesuffix('.py') if you have python>=3.9
        if file_.endswith('.py'): file_ = file_[:-3]    
    default_configs = getattr(importlib.import_module(file_), class_)
    for key in new_cfg:
        if new_cfg[key] is None: continue
        my_setattr(conf_class=default_configs, key=key, new_value=new_cfg[key], vb=vb)
    altered_cv = secure_chained_vars(default_configs, new_cfg, vb)
    if vb:
        print绿(''.join(['-']*len(str_pro)),)
        arg_summary(default_configs, new_cfg, altered_cv)
        print绿(''.join(['-']*len(str_pro)),'\n\n\n')
    if 'TEAM_NAMES' in new_cfg:
        return [item.split('->')[0] for item in new_cfg['TEAM_NAMES']]
    return None
        
def check_config_relevence(json_data):
    env_name = json_data['config.py->GlobalConfig']['env_name']
    env_path = json_data['config.py->GlobalConfig']['env_path']
    for key in json_data.keys():
        if 'MISSION' in key: assert env_path in key, ('configering wrong env!')

def load_config_via_json(json_data, vb):
    for cfg_group in json_data:
        dependency = override_config_file(cfg_group, json_data[cfg_group], vb)
        if dependency is not None:
            for dep in dependency:
                assert any([dep in k for k in json_data.keys()]), 'Arg check failure, There is something missing!'
    check_config_relevence(json_data)
    return None

def get_core_args(vb=True):
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-c', '--cfg', help='Path of the configuration file')
    parser.add_argument('-s', '--skip', action='store_true', help='skip logdir check')
    args, unknown = parser.parse_known_args()
    load_via_json = (hasattr(args, 'cfg') and args.cfg is not None)
    if load_via_json:
        if len(unknown) > 0 and vb:  
            print亮红('Warning! In json setting mode, %s is ignored'%str(unknown))
        import commentjson as json
        with open(args.cfg) as f: json_data = json.load(f)
        core_group = 'config.py->GlobalConfig'
        override_config_file(core_group, json_data[core_group], vb)
    else:
        assert False

    from config import GlobalConfig as cfg
    return cfg

def prepare_recycle_folder():
    import glob
    if not os.path.exists('./TEMP'): os.mkdir('./TEMP')
    for tmp in glob.glob('./TEMP/find_free_ports_*'):
        os.remove(tmp)

def prepare_args(vb=True):
    prepare_recycle_folder()
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-c', '--cfg', help='Path of the configuration file')
    parser.add_argument('-s', '--skip', action='store_true', help='skip logdir check')
    args, unknown = parser.parse_known_args()
    load_via_json = (hasattr(args, 'cfg') and args.cfg is not None)
    assert load_via_json
    skip_logdir_check = (hasattr(args, 'skip') and (args.skip is not None) and args.skip) or (not vb)

    if len(unknown) > 0 and vb: 
        print亮红('Warning! In json setting mode, %s is ignored'%str(unknown))
    import commentjson as json
    with open(args.cfg, encoding='utf8') as f:
        json_data = json.load(f)
    new_args = load_config_via_json(json_data, vb)

    from config import GlobalConfig as cfg
    note_name_overide = None
    if not skip_logdir_check: 
        note_name_overide = check_experiment_log_path(cfg.logdir)
        if note_name_overide is not None: 
            override_config_file('config.py->GlobalConfig', {'note':note_name_overide}, vb)
    if not os.path.exists(cfg.logdir): os.makedirs(cfg.logdir)
    if not cfg.recall_previous_session: 
        copyfile(args.cfg, '%s/experiment.json'%cfg.logdir)
        backup_files(cfg.backup_files, cfg.logdir)
        cfg.machine_info = register_machine_info(cfg.logdir)
    cfg.cfg_ready = True
    return cfg

def register_machine_info(logdir):
    import socket, json, subprocess, uuid
    from .network import get_host_ip
    info = {
        'HostIP': get_host_ip(),
        'ExpUUID':uuid.uuid1().hex,
        'RunPath': os.getcwd(),
        'StartDateTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    try:
        info['DockerContainerHash'] = subprocess.getoutput(r'cat /proc/self/cgroup | grep -o -e "docker/.*"| head -n 1 |sed "s/docker\\/\\(.*\\)/\\1/" |cut -c1-12')
    except: 
        info['DockerContainerHash'] = 'None'
    with open('%s/info.json'%logdir, 'w+') as f:
        json.dump(info, f, indent=4)
    return info

def backup_files(files, logdir):
    for file in files:
        if os.path.isfile(file):
            print绿('[config] Backup File:',file)
            bkdir = '%s/backup_files/'%logdir
            if not os.path.exists(bkdir): os.makedirs(bkdir)
            copyfile(file, '%s/%s'%(bkdir, os.path.basename(file)))
        else:
            print亮绿('[config] Backup Folder:',file)
            assert os.path.isdir(file), ('cannot find', file)
            copytree(file, '%s/backup_files/%s'%(logdir, os.path.basename(file)), 
                dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))

    return 

def check_experiment_log_path(logdir):
    res = None
    if os.path.exists(logdir):
        if os.path.exists(logdir+'test_stage'): return None
        print亮红('Current log path:', logdir)
        print亮红('Warning! you will overwrite old logs if continue!')
        print亮红("Pause for 60 seconds before continue (or press Enter to confirm!)")
        try:
            res = askChoice()
            if res == '': res = None
        except func_timeout.exceptions.FunctionTimedOut as e:
            res = None
    return res


@func_timeout.func_set_timeout(60)
def askChoice():
    return input('>>')



# def input_or_timeout(timeout):
#     for i in range(30):
#         time.sleep(1)
#     return

def arg_summary(config_class, modify_dict = {}, altered_cv = []):
    for key in config_class.__dict__: 
        if '__' in key: continue
        if key.endswith('_cv'): continue
        if (not key in modify_dict) or (modify_dict[key] is None): 
            if key not in altered_cv: 
                print绿(key.center(25), '-->', str(getattr(config_class,key)))
            else: 
                print靛(key.center(25), '-->', str(getattr(config_class,key)))
        else: 
            print红(key.center(25), '-->', str(getattr(config_class,key)))


def my_setattr(conf_class, key, new_value, vb):
    assert hasattr(conf_class, key), (conf_class, 'has no such config item: **%s**'%key)
    setting_name = key
    replace_item = new_value
    original_item = getattr(conf_class, setting_name)
    if vb: print绿('[config] override %s:'%setting_name, original_item, '-->', replace_item)
    if isinstance(original_item, float):
        replace_item = float(replace_item)
    elif isinstance(original_item, bool):
        if replace_item == 'True':
            replace_item = True
        elif replace_item == 'False':
            replace_item = False
        elif isinstance(replace_item, bool):
            replace_item = replace_item
        else:
            assert False, ('enter True or False, but have:', replace_item)
    elif isinstance(original_item, int):
        assert int(replace_item) == float(replace_item), ("warning, this var **%s** has an int default, but given a float override!"%key)
        replace_item = int(replace_item)
    elif isinstance(original_item, str):
        replace_item = replace_item
    elif isinstance(original_item, list):
        assert isinstance(replace_item, list)
    elif isinstance(original_item, dict):
        assert isinstance(replace_item, dict)
    else:
        assert False, ('not support this type')
    setattr(conf_class, setting_name, replace_item)
    return

def find_all_conf():
    import glob
    py_script_list = glob.glob('./**/*.py', recursive=True)
    conf_class_gather = []
    for python_file in py_script_list:
        with open(python_file,encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            if 'ADD_TO_CONF_SYSTEM' not in line: continue
            if 'class ' not in line: continue
            conf_class_gather.append({'line':line, 'file':python_file})
    def getBetween(str, str1, str2):
        strOutput = str[str.find(str1)+len(str1):str.find(str2)]
        return strOutput
    for target in conf_class_gather:
        class_name = getBetween(target['line'], 'class ', '(')
        target['class_name'] = class_name
        target['file'] = target['file'].replace('/', '.').replace('..', '')
        import importlib
        target['class'] = getattr(importlib.import_module(target['file'].replace('.py', '')), class_name)
    return conf_class_gather

def make_json(conf_list):
    import json
    out = {}
    for conf in conf_list:
        local_conf = {}
        config_class = conf['class']
        for key in config_class.__dict__: 
            if '__' in key or '_cv' in key: continue
            item_to_be_serialize = getattr(config_class, key)
            try:
                json.dumps(item_to_be_serialize)
            except:
                item_to_be_serialize = '[cannot be json]' + str(item_to_be_serialize)
            local_conf[key] = item_to_be_serialize
        out[conf['file']] = local_conf
    # json_str = json.dumps(out)
    with open('all_conf.json', 'w') as f:
        json.dump(out, f, indent=4)
        print亮紫('the conf summary is successfully saved to all_conf.json')

if __name__ == '__main__':
    conf_list = find_all_conf()
    res_json = make_json(conf_list)