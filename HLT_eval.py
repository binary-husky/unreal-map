import os, shutil, subprocess, glob, re, threading, time
import commentjson as json
import numpy as np
from UTIL.colorful import print亮紫, print亮绿




def eval_with_config(target_json, note_name, cpk, test_which_cpk, hete_n_alive_frontend):
    with open(target_json, encoding='utf8') as f:
        json_data = json.load(f)

    json_data["config.py->GlobalConfig"]["note"] = note_name
    json_data["config.py->GlobalConfig"]["test_only"] = True
    json_data["config.py->GlobalConfig"]["device"] = 'cuda'
    json_data["config.py->GlobalConfig"]["report_reward_interval"] = 512
    # 
    json_data["ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig"]["load_checkpoint"] = True
    json_data["ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig"]["load_specific_checkpoint"] = cpk
    #
    json_data["ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig"]["hete_n_alive_frontend"] = hete_n_alive_frontend
    json_data["ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig"]["policy_matrix_testing"] = True
    json_data["ALGORITHM.hete_league_onenet_fix.foundation.py->AlgorithmConfig"]["test_which_cpk"] = test_which_cpk
    with open(target_json, 'w') as f:
        json.dump(json_data, f, indent=4)


    matrix_log_dir = os.path.join(os.path.dirname(target_json), 'matrix')
    if not os.path.exists(matrix_log_dir): os.makedirs(matrix_log_dir)
    cpk_num = cpk.split('_{\'win_rate\'')[0].split('/model_')[1]
    f = open(os.path.join(matrix_log_dir,'_c%d_a%d_m%d'%(test_which_cpk, hete_n_alive_frontend, int(cpk_num))),'w')
    print(target_json, cpk, test_which_cpk, hete_n_alive_frontend)
    subp = subprocess.Popen([
        'python', 'main.py', '-c', target_json, '--skip'
    ], stdout=f)
    subp.wait()
    f.close()
# target_json = 'ZHECKPOINT/noHLT/experiment_test.jsonc'
# cpk = "history_cpt/model_4037_{'win_rate': 0.9140625, 'mean_reward': 1.5679687500000001}.pt"

# note_name = 'HLT-L2-cos-run1'
# cpk = "history_cpt/model_3507_{'win_rate': 0.984375, 'mean_reward': 1.8031249999999999}.pt"

# note_name = 'HLT-L2-cos-run2'
# cpk = "history_cpt/model_3537_{'win_rate': 0.984375, 'mean_reward': 1.794921875}.pt"
note_list = [
    # "alive1-sin-run1",
    # "alive1-sin-run2",
    # "prob0d2-cos-run1",
    # "prob0d2-cos-run2",
    # "prob0d2-cos-run3",

    "prob0d2-cos-run4",
    "prob0d2-cos-run5",
    "prob0d2-cos-run6",
    # "NoHLT-cos-run4",
    # "NoHLT-cos-run5",
    # "NoHLT-cos-run6",
    # "NoHLT-cos-run7",

]

def check_already_done(target_json, note_name, cpk, test_which_cpk, hete_n_alive_frontend):
    target_json = 'ZHECKPOINT/%s/experiment_test.jsonc'%note_name
    target_dir = 'ZHECKPOINT/%s/matrix'%note_name

    search_res = glob.glob(target_dir+'/*')
    data = []
    matrix_log_dir = os.path.join(os.path.dirname(target_json), 'matrix')
    cpk_num = cpk.split('_{\'win_rate\'')[0].split('/model_')[1]
    p = os.path.join(matrix_log_dir,'_c%d_a%d_m%d'%(test_which_cpk, hete_n_alive_frontend, int(cpk_num)))
    # print(p)
    if not os.path.exists(p): return False
    with open(p,'r') as f:
        ls = [line for line in f.readlines() if 'agents of interest: ' in line]
        if len(ls)>0:
            return True
        else:
            return False



def do(note_name):
    target_dir = 'ZHECKPOINT/%s/history_cpt'%note_name
    search_res = glob.glob(target_dir+'/*.pt')
    win_rate_max = -1
    win_rate_max_index = -1
    wr = []
    for i, cpt in enumerate(search_res):
        baseName = os.path.basename(cpt)
        res = re.findall(
                re.compile(r"win_rate': (.*?),"), baseName
            )[0]
        win_rate = float(res)
        wr.append(win_rate)
        # if win_rate >= win_rate_max:
        #     win_rate_max = win_rate
        #     win_rate_max_index = i
    wr_index = np.argsort(wr)
    cpk = 'history_cpt' + search_res[wr_index[-3]].split('history_cpt')[1]
    target_json_runtime = 'ZHECKPOINT/%s/experiment.jsonc'%note_name
    target_json = 'ZHECKPOINT/%s/experiment_test.jsonc'%note_name
    shutil.copyfile(target_json_runtime, target_json)


    for test_which_cpk in range(1, 4+1):
        max_alive_frontend = 2
        if test_which_cpk==1:
            max_alive_frontend=3
        # else:
        #     continue
        for hete_n_alive_frontend in range(0, max_alive_frontend+1):
            print亮紫(target_json, note_name, cpk, test_which_cpk, hete_n_alive_frontend)
            if check_already_done(target_json, note_name, cpk, test_which_cpk, hete_n_alive_frontend): 
                print亮绿('skip')
                continue
            threading.Thread(target=eval_with_config, args=(target_json, note_name, 
                cpk, test_which_cpk, hete_n_alive_frontend,)).start(); time.sleep(5)
            print亮绿('commited')

for note_name in note_list:
    do(note_name)

