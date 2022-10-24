# %matplotlib inline
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def setTimesNewRomanFont_MustExecuteAtLast():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] # + plt.rcParams['font.serif']

def init(font_scale):
    sns.set_theme(
        context='paper',    # notebook, paper, talk, poster
        style='whitegrid',  # darkgrid, whitegrid, dark, white, ticks
        palette='deep',     # https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
        font='sans-serif', 
        font_scale=font_scale, 
        color_codes=True, 
        rc=None)
    setTimesNewRomanFont_MustExecuteAtLast()

def roll_color_palette(cp, offset):
    pre = cp[:offset]
    post = cp[offset:]
    return post + pre

def lift_color(cp, n):
    return [ cp[n] ] + [ c for i, c in enumerate(cp) if i!=n ]

def find_in_dict_list(dict_list, **kwargs):
    res = None
    for d in dict_list:
        if all([d[k] == v for k, v in kwargs.items()]): 
            res = d
            break
    return res

def filter_in_dict_list(dict_list, **kwargs):
    res = []
    for d in dict_list:
        if all([d[k] == v for k, v in kwargs.items()]): 
            res.append(d)
    return res

def filter_out_dict_list(dict_list, **kwargs):
    res = []
    for d in dict_list:
        if all([d[k] == v for k, v in kwargs.items()]): 
            pass
        else:
            res.append(d)
    return res

def rename_key_in_dict_list(dict_list, from_what, to_what):
    res = []
    for d in dict_list:
        for k, v in d.items():
            if v == from_what: d[k] = to_what
        res.append(d)
    return res

def lift_key_in_dict_list(dict_list, key):
    res = []
    for d in dict_list:
        if any([v == key for k, v in d.items()]):
            res.append(d)

    for d in dict_list:
        if not any([v == key for k, v in d.items()]):
            res.append(d)
    return res

# 左下角为0点
def legend(handle, 水平位置百分比, 垂直位置百分比, 边框):
    # https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib/39806180#39806180
    sns.move_legend(
        handle, "center", frameon = 边框,
        bbox_to_anchor=(水平位置百分比, 垂直位置百分比)
    )

def save_and_push(handle, img_path, check_exist=True):
    if check_exist and os.path.exists(img_path): 
        assert False, "image already exists!"
    handle.savefig(img_path, bbox_inches='tight')
    image_basename = os.path.basename(img_path)
    subprocess.Popen([
        'curl', '-T', img_path, '-u', 'fuqingxu:clara', 
        'http://cloud.fuqingxu.top:4080/remote.php/dav/files/fuqingxu/research2/heteGrouping/img/%s'%image_basename
    ])
''' ratio problem 1

g = sns.pointplot(x="N Agents", y="Average Test Reward", hue="Method", data=data, 
    aspect=2.7 , height=5, 
    capsize=.35, kind="bar", palette =sns.color_palette("pastel") ,legend_out=True)

'''

# sns.move_legend(
#     res, "lower left",
#     bbox_to_anchor=(0.68, 0.55)
# )
# changedNameOfImage = True


# nameOfImage = "ADCA-Two-Phase"
# path = "./imgsave/"
# assert changedNameOfImage
# plt.savefig('%s/%s.pdf'%(path,nameOfImage),bbox_inches='tight')

# curl -T ./imgsave/ADCA-Two-Phase.pdf -u fuqingxu:clara http://cloud.fuqingxu.top:4080/remote.php/dav/files/fuqingxu/research/paper03_phase3/DoR-LMAS/img/ADCA-Two-Phase.pdf


'''

# %load_ext autoreload
# %autoreload 3
import os, shutil, subprocess, glob, re
import commentjson as json
import seaborn as sns
import pandas as pd
import matplotlib


note_list = [
    "NoHLT-cos-run3",
    "NoHLT-cos-run4",
    "NoHLT-cos-run5",
    "NoHLT-cos-run6",
    "NoHLT-cos-run7",

    "prob0d2-cos-run1",
    "prob0d2-cos-run2",
    "prob0d2-cos-run3",
]

data = []
for note_name in note_list:
    target_json = 'ZHECKPOINT/%s/experiment_test.jsonc'%note_name
    target_dir = 'ZHECKPOINT/%s/matrix'%note_name
    method = note_name.split('-')[0] + '-'+ note_name.split('-')[-2]
    print(method)
    search_res = glob.glob(target_dir+'/*')
    for p in (search_res):
        base_name = os.path.basename(p)
        res = base_name.split('_')
        which_ckp = int(res[1].split('c')[1])
        alive_frontier = int(res[2].split('a')[1])
        update_cnt = int(res[3].split('m')[1])
        # print(p)
        with open(p,'r') as f:
            in_line = [line for line in f.readlines() if 'agents of interest: ' in line][0]

        res = re.findall(
                re.compile(r"recent reward (.*?), best reward (.*?), win rate (.*?)$"), in_line
            )[0]
        reward = float(res[0])
        win_rate = float(res[2])
        data.append(
        {
            'note_name':note_name,
            'method':method,
            'which_ckp':which_ckp,
            'alive_frontier':alive_frontier,
            'update_cnt':update_cnt,
            'reward':reward,
            'win_rate':win_rate,
            'target_dir':target_dir,
        }
        )

    def find_in_dict(dict_list, **kwargs):
        res = None
        detail_debug = []
        for d in dict_list:
            detail_debug.append([d[k] == v for k, v in kwargs.items()])
            if all([d[k] == v for k, v in kwargs.items()]): 
                res = d
                break
        return res
    # print(data)
    frontier_win_rate = res = find_in_dict(data, which_ckp=1, alive_frontier=3, method=method, target_dir=target_dir)['win_rate']
    # print('frontier win rate', res['win_rate'])

    for test_which_cpk in range(1,5):
        # print('test_which_cpk', test_which_cpk, end='\t')
        for alive in range(3):
            res = find_in_dict(data, which_ckp=test_which_cpk, alive_frontier=alive)
            # print(res['win_rate'], end='\t')
        # print('')



    for test_which_cpk in range(1,5):
        # print('test_which_cpk', test_which_cpk, end='\t')
        for alive in range(3):
            base_line = find_in_dict(data, which_ckp=test_which_cpk, alive_frontier=0, method=method, target_dir=target_dir)
            res = find_in_dict(data, which_ckp=test_which_cpk, alive_frontier=alive, method=method, target_dir=target_dir)
            res['baseline'] = float(base_line['win_rate'])
            res['inc'] = (float(res['win_rate'])-float(base_line['win_rate']))# /frontier_win_rate
            # print(res['inc'], end='\t')
        # print('')

# %matplotlib inline
# ! rm /home/hmp/.cache/matplotlib -rf

from VISUALIZE.seaborn_defaults import *
init(font_scale=1.7)
data_p = filter_in_dict_list(data, alive_frontier=2)
data_p = rename_key_in_dict_list(data_p, from_what='NoHLT-cos', to_what='without HLT')
data_p = rename_key_in_dict_list(data_p, from_what='prob0d2-cos', to_what='HLT')
data_p = lift_key_in_dict_list(data_p, key='HLT')

data_p = pd.DataFrame(data_p)


cp = sns.color_palette("husl")
cp = roll_color_palette(cp, offset=4)
cp = lift_color(cp, n=1)

sns.set_palette(cp)

g = sns.lmplot(
    data=data_p,
    x="baseline", y="inc", hue="method",
    aspect=1.27
)
g.set_axis_labels("Past Policy Win Rate", "Improvement Score")
legend(g, 水平位置百分比=0.323, 垂直位置百分比=0.29, 边框=True)
plt.savefig('temp.jpg', bbox_inches='tight')


'''

