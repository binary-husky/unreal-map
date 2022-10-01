import os, shutil, subprocess
import commentjson as json




def eval_with_config(target_json, cpk, test_which_cpk, hete_n_alive_frontend):
    with open(target_json, encoding='utf8') as f:
        json_data = json.load(f)

    json_data["config.py->GlobalConfig"]["test_only"] = True
    json_data["config.py->GlobalConfig"]["device"] = 'cuda:1'
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

target_json = 'ZHECKPOINT/noHLT/experiment_test.jsonc'
cpk = "history_cpt/model_4037_{'win_rate': 0.9140625, 'mean_reward': 1.5679687500000001}.pt"
eval_with_config(target_json, cpk, 1, 3)
for test_which_cpk in range(1, 4+1):
    for hete_n_alive_frontend in range(0, 2+1):
        eval_with_config(target_json, cpk, test_which_cpk, hete_n_alive_frontend)
