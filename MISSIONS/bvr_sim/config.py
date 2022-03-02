"""
@FileName：config.py
@Description：
@Author：wubinxing
@Time：2021/5/9 下午8:08
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
# from agent.origin_demo_agent import DemoAgent
# from agent.new_agent import NewAgent
# from agent.new_agent2 import NewAgent as NewAgent2 # Team_Name_Orig
# from agent.Team_Name_Red.Team_Name import  Team_Name_Policy as attack_vip
# from agent.Team_Name.Team_Name import  Team_Name_Policy as strike
# from agent.cm.Cm import Cm as simple
# from agent.Cm5.Cm5 import Cm5 as Cm5
# from agent.Yi_team.Yi_team import Yi_team as Yi_team_v5
# from agent.Yi_team_v1.Yi_team import Yi_team as vip_escape
# from agent.Yi_team_v2.Yi_team import Yi_team as vip_2x
# from agent.Team_Name_Orig.Team_Name import  Team_Name_Policy as strike_blue
# from agent.debug_red import DebugAgentRed
# from agent.demo_agent import DemoAgent
# from agent.Yi_team_test_esc.Yi_team import Yi_team as Yi_team_test_esc
# from agent.Yi_team_new.Yi_team import Yi_team as Yi_team_v6
# from agent.Cm5_super0.Cm5 import Cm5 as Cm5_super
# from agent.Team_Name_Red.Team_Name import  Team_Name_Policy as DebugAgentRed
# from agent.Team_Name_Blue.Team_Name import  Team_Name_Policy as DebugAgentBlue
# from agent.量子飞机v2.Yi_team import Yi_team as 量子飞机
# from agent.Team_Name_Red_Full_Attack.Team_Name import  Team_Name_Policy as DebugAgentRed
from agent.cm_new.Cm import Cm as cm_new
# from agent.Yi_team_submit.Yi_team import Yi_team as Yi_team_old
# from agent.Cm5.Cm5 import Cm5 as cm_old
# from agent.Yi_team_fusai_test_bug.Yi_team import Yi_team
# from agent._0_yiteam_final_v1.Yi_team import Yi_team
# from agent._0_yiteam_final_v2.Yi_team import Yi_team as Yi_team_ref
# from agent._0_yiteam_final_v3_c1.Yi_team import Yi_team
# from agent.Yi_team_fusai_V3.Yi_team import Yi_team
# from agent.super_agent.main import Super_agent
# from agent.super_agent_new.main import Super_agent
# from agent.Yi_team_fusai_V4.Yi_team import Yi_team as Yi_team_old
# from agent.yiteam_twin_test.Yi_team import Yi_team as Yi_team_twin
# from agent.super_agent_new2.main import Super_agent

from agent.yiteam_final_commit_v7.Yi_team import Yi_team

ISHOST = True

# 为态势显示工具域分组ID  1-1000
HostID = 1

IMAGE = 'fuqingxu/bvrsim:trim'

# 加速比 1-100
TimeRatio = 100

# 范围:0-100 生成的回放个数 (RTMNum + 2),后续的回放会把之前的覆盖掉.
RTMNum = 0

config = {
    "episode_time": 20000000,   # 训练次数
    "step_time": 1, # 想定步长
    'agents': {
        'red':Yi_team , # Super_agent, #  Yi_team_twin, #Super_agent  ,
        'blue': Yi_team   #attack_vip # Cm5 # Yi_team_ref # Yi_team #Super_agent
    }
}

# 进程数量
POOL_NUM = 10

# 启动XSIM的数量
XSIM_NUM = 6

def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

ADDRESS = {
    "ip": "127.0.0.1",
    "port": find_free_port()
}
