#coding=utf-8
import glob,os,sys,re,subprocess,platform


def print红(*kw):
    print("\033[0;31m",*kw,"\033[0m")
def print绿(*kw):
    print("\033[0;32m",*kw,"\033[0m")
def print黄(*kw):
    print("\033[0;33m",*kw,"\033[0m")
def print蓝(*kw):
    print("\033[0;34m",*kw,"\033[0m")
def print紫(*kw):
    print("\033[0;35m",*kw,"\033[0m")
def print靛(*kw):
    print("\033[0;36m",*kw,"\033[0m")
def printX(*kw):
    print("\033[0;38m",*kw,"\033[0m")

# 用pip执行安装指令
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
        "-i","https://pypi.tuna.tsinghua.edu.cn/simple",
        "--progress-bar","emoji",
        "--prefer-binary", 
        package])
    except:
        print红("执行命令 ", "pip", "install","-i","https://pypi.tuna.tsinghua.edu.cn/simple", package, "时，抛出错误")
        pass

sys_name = platform.system()
if sys_name == "Windows":
    try:
        from colorama import init,Fore,Back,Style
        init(autoreset=False)
        def print红(*kw):
            print(Fore.RED,*kw)
        def print绿(*kw):
            print(Fore.GREEN,*kw)
        def print黄(*kw):
            print(Fore.YELLOW,*kw)
        def print蓝(*kw):
            print(Fore.BLUE,*kw)
        def print紫(*kw):
            print(Fore.MAGENTA,*kw)
        def print靛(*kw):
            print(Fore.CYAN,*kw)

    except:
        install('colorama')
        print('颜色组件安装完成！现在请重新运行！')
        sys_name.exit(0)

"""
# step 1, 查询所有子路径.py脚本文件，列表
"""
py_script_list = glob.glob('./**/*.py', recursive=True)
required = []
local_name_list = {"None":False}
引发连锁错误的包_列表 = {"None":False}


"""
# step 2, 提取 import 以及 from *** import
"""
def 是否为工程内的文件交叉调用(包,python_file):
    包_org = 包
    if '.' not in 包:
        res = os.path.exists("./"+包+".py")
        if res:
            return True,包_org
        else:
            return False, 包_org
    if 包.startswith('.'):
        包 = os.path.dirname(python_file).replace("/", ".").replace("..", ".") + 包
        包_org = 包
    包 = 包.replace(".", "/")
    res = os.path.exists("./"+包+".py")
    if res:
        tmp = 包_org.split(".")
        if tmp[0]!='': local_name_list[tmp[0]] = True
        return True, 包_org
    else:
        return False, 包_org

for python_file in py_script_list:
    with open(python_file,encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            if "import" in line or "from" in line:
                t = line.split()
                # from 开头 或者 import开头
                if t[0] == "import" or t[0] == "from":
                    i = 1
                    包 = ""
                    for ti in t[1:]:
                        if (ti!="import") and (ti!="as"):
                            包 = 包 + ti
                        else:
                            break
                    if "," in 包:
                        包_l = 包.split(",")
                    else:
                        包_l = [包]
                    for 包 in 包_l:
                        包_debug = 包
                        if 包_debug == '.':
                            continue
                        res,包 = 是否为工程内的文件交叉调用(包,python_file)
                        if not res:
                            required.append(包)
                            


required = set(required)
required = sorted(required)

"""
# step 3, 尝试import，筛查缺失的包
"""
print黄("**************************************************************")
print黄("尝试import")
# 使用清华镜像
need_fix_cmd_orig = "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
need_fix_cmd      = "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
need_fix_list = []
failed_cmd = []
chain_failed = []
for 包 in required:
    cmd = "import "+包
    try:
        # 如果这里罕见地报错，
        # 说明该文件有 import开头的、被“”“包裹的注释, 
        # 找到它，然后删除这个奇葩注释
        exec(cmd)
    except ImportError as error:
        print红("error trying to do:",cmd,error.msg)
        error_str = error.msg.split('\'')
        package_import_error = (len(error_str) >= 2)
        if not package_import_error:
            continue
        包_error = error_str[1]
        if '.' in 包:
            包_l = 包.split('.')
            包_tmp = 包_l[0]
        # 引发问题的不是这个包本身，而是这个包import其他包，但这个包内引包失败了
        # 仅仅是一个连锁错误而已，无需处理
        if 包_error != 包:
            print红("发生连锁引包错误: ",error.msg)
            chain_failed.append(error.msg)
            cmd = cmd + "\t\t此项仅仅由连锁引包错误导致: " + error.msg
            if 包_tmp not in 引发连锁错误的包_列表:
                引发连锁错误的包_列表[包_tmp]=True
        else:
            # 非连锁，一定是真的缺
            引发连锁错误的包_列表[包]=False
        failed_cmd.append(cmd)
        if '.' in 包:
            包_l = 包.split('.')
            包 = 包_l[0]
        if len(包)>19: # some comment mixed in somehow
            continue
        need_fix_list.append(包)
    except BaseException as error:
        print红(error)
    else:
        print绿("this package is ok:",cmd)

need_fix_list = set(need_fix_list)
need_fix_list = sorted(need_fix_list)
if len(failed_cmd) > 0:
    print红("以下的包import操作失败")
    for cmd in failed_cmd:
        print红(cmd)


"""
# step 4, 处理缺失的包，并找到对应的pip安装指令
"""
term_replace_dict = {
    "cv2":"opencv-python",
    "torch":"torch",
    "mpi":"mpi4py",
    "MPI":"mpi4py",
    "mujoco_py":"None",    # pip cannot install this????
    "pybullet_envs":"None",
    "stable_baselines3":"None",
    "pyximport":"cython",
    "PIL":"None",
    "collective_assult":"None",
    "gym_fortattack":"None",
    "multiagent":"None",
    "z_config":"None",
    "gym_vecenv":"None"
}

PIL

for inx, 包 in enumerate(need_fix_list):
    if 包 in term_replace_dict:
        包 = term_replace_dict[包]
        need_fix_list[inx] = 包
    if (包 in local_name_list) or (包 in 引发连锁错误的包_列表 and 引发连锁错误的包_列表[包]==True):
        need_fix_list[inx] = "None"

need_fix_list = set(need_fix_list)
need_fix_list = sorted(need_fix_list)
if len(need_fix_list) == 0:
    print绿("所有依赖已就绪")
    exit(0)

"""
# step 5, 如果有requirement.txt，从中提取出有用的版本信息
"""
print黄("**************************************************************")
print蓝("requirement.txt中的相关信息")
execute_fix = []
if os.path.exists("./requirements.txt"):
    with open("./requirements.txt",encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("-"):
                print蓝("requirement.txt要求以下版本： -->   "+"pip install "+line[:-1])
                print蓝("首先git clone，然后找到setup.py的路径，然后执行 pip install --no-deps  -e .")
                continue
            line_split = line.split("==")
            if (len(line_split)==2) and (line_split[0] in need_fix_list):
                print蓝("requirement.txt要求以下版本： -->   "+"pip install "+line[:-1])



"""
# step 6, 如果需要安装pytorch，gym等特殊包，对应给出安装建议
"""

def config_anaconda():
    with open(__file__,'r') as f:
        conda_cmd = f.readlines()
        condarc_lines = conda_cmd[-18:-2]
    f = open('./.condarc','w+')
    f.writelines(condarc_lines)
    f.close()


print黄("**************************************************************")
try:
    conda_env_name = sys.executable.split('/')[-3]
except:
    conda_env_name = sys.executable.split('\\')[-2]
for 包 in need_fix_list:
    if 包 == "torch":
        print蓝("pytorch需要手动安装，pytorch 的安装方法（选择其一），然后重新运行该脚本:")
        print蓝("conda install -n %s  pytorch torchvision torchaudio cudatoolkit=10.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/"%conda_env_name)
        print蓝("conda install -n %s  pytorch torchvision torchaudio cudatoolkit=11.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/"%conda_env_name)
        print蓝("")
        
        sys.exit(0)
    # if 包 == "tensorflow":
    #     print靛("Tensorflow需要手动安装，首先，更换conda源的指令")
    #     config_anaconda()
    #     print靛("cp",os.getcwd()+"/.condarc","~/.condarc")
    #     print靛("然后，安装TF一代的指令")
    #     print靛("conda install -n %s tensorflow-gpu=1.*"%conda_env_name)
    #     sys.exit(0)
    if (包 is not "None"):
        need_fix_cmd = need_fix_cmd + 包 + "  "
        execute_fix.append(包)
print黄("**************************************************************")
print绿(need_fix_cmd)
print黄("**************************************************************")




"""
# step 7, 对于除了特殊包之外的其他软件包，调用pip直接安装
"""
print绿("注意！当前的conda环境是：",conda_env_name," 所有操作都将只在该conda环境内生效")
input("执行自动安装？")
if input("确定执行自动安装？(y/n)")=='y':
    for 包 in execute_fix:
        install(包)


"""
# step 8, 完成任务，取消以下代码的注释，测试pytorch是否工作
"""
        
# import torch
# flag = torch.cuda.is_available()
# print(flag)

# ngpu= 1
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print(device)
# print(torch.cuda.get_device_name(0))
# print(torch.rand(3,3).cuda()) 

'''
不要修改或者删除以下内容！！有用！！
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
'''