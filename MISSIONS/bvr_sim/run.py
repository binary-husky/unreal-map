
from agent.yiteam_final_commit_v7.UTILS.colorful import print亮红
from config import ADDRESS, config, POOL_NUM, ISHOST, XSIM_NUM
from ..env.xsim_env import XSimEnv
from ..env.env_runner import EnvRunner
from time import sleep
from multiprocessing import Pool
import time

# 启动单个XSIM
class BattleRunnerSignal(EnvRunner):
    """
        对战环境
        @Examples:
        @Author:wubinxing
        """
    def __init__(self, agents, address, mode='host'):
        EnvRunner.__init__(self, agents, address, mode)  # 仿真环境初始化

    def run(self, num_episodes):
        self._end(); 
        print亮红('press any key to continue')
        input()
        self._reset()
        map_start = 'y' # input("是否已启动态势显示工具? Y or N")
        start = 'y'  # input("是否开始运行? Y or N")
        battle_results = [0,0,0]
        if start == "Y" or start == "y":
            # [红方获胜局数, 蓝方获胜局数, 平局数量]
            if map_start == "Y" or map_start == "y":


                for i in range(num_episodes):
                # print("重置智能体与环境并获取obs!!!!!!!!!!!=====>", i)
                   # 重置智能体与环y
                # 境并获取obs
                    obs = self.step([])
                    # print("obs:", obs)
                    while True:
                        if obs is None:
                            sleep(1)
                            obs = self.step([])
                            continue
                        done = self.get_done(obs)  # 推演结束(分出胜负或达到最大时长)
                        #print("done:", done)
                        if done[0]:  # 对战结束后环境重置
                            with open('debug_profile.txt', 'a+') as f:
                                f.write('Dead\n')
                            print("到达终止条件!!!!")
                            battle_results[0] += int(done [1]==1 and done[2]==0)
                            battle_results[1] += int(done [1]==0 and done[2]==1)
                            battle_results[2] += int((done [1]==1 and done[2]==1) or (done [1]==0 and done[2]==0))
                            if(done [1]==1 and done[2]==0):
                                print("第",i + 1,  "局 :  红方胜")
                            elif(done [1]==0 and done[2]==1):
                                print("第", i + 1, "局:   蓝方胜")
                            else:
                                print("第", i + 1, "局:   平  局")
                            break
                        obs = self.run_env(obs)

                        # except Exception as e:
                        #     print(e)
                        #     print("运行出现异常需要重启")
                        #     break

                    # sleep(20)
                    print("共", i + 1, "局:   红方胜", battle_results[0], "局:   蓝方胜", battle_results[1], "局:   平局",battle_results[2], "局")
                    self._end()
                    print("请点击态势显示工具的实时态势按钮或者快捷键F2!")
                    input()
                    reset = 'y'
                    print("重置运行:", reset)
                    if reset == "Y" or reset == "y":
                        self._reset()
                    else:
                        break
                    print(i + 1, "重置")
            else:
                for i in range(num_episodes):
                    obs = self.step([])
                    while True:
                        if obs is None:
                            sleep(1)
                            obs = self.step([])
                            continue
                        done = self.get_done(obs)  # 推演结束(分出胜负或达到最大时长)
                        #print("done:", done)
                        if done[0]:  # 对战结束后环境重置
                            print("到达终止条件!!!!")
                            battle_results[0] += int(done [1]==1 and done[2]==0)
                            battle_results[1] += int(done [1]==0 and done[2]==1)
                            battle_results[2] += int((done [1]==1 and done[2]==1) or (done [1]==0 and done[2]==0))
                            if(done [1]==1 and done[2]==0):
                                print("第",i + 1,  "局 :  红方胜")
                            elif(done [1]==0 and done[2]==1):
                                print("第", i + 1, "局:   蓝方胜")
                            else:
                                print("第", i + 1, "局:   平  局")
                            break
                        obs = self.run_env(obs)
                    print("共", i + 1, "局:   红方胜", battle_results[0], "局:   蓝方胜", battle_results[1], "局:   平局",battle_results[2], "局")
                    self._reset()
                    print(i, "重置")
            return battle_results

    def run2(self, num_episodes):
        # [红方获胜局数, 蓝方获胜局数, 平局数量]
        battle_results = [0, 0, 0]
        for i in range(num_episodes):
            obs = self.step([])
            while True:
                if obs is None:
                    sleep(1)
                    obs = self.step([])
                    continue
                done = self.get_done(obs)  # 推演结束(分出胜负或达到最大时长)
                if done[0]:  # 对战结束后环境重置
                    print("到达终止条件!!!!")
                    battle_results[0] += int(done [1]==1 and done[2]==0)
                    battle_results[1] += int(done [1]==0 and done[2]==1)
                    battle_results[2] += int((done [1]==1 and done[2]==1) or (done [1]==0 and done[2]==0))
                    if done [1]==1 and done[2]==0:
                        print("第",i + 1,  "局:  红方胜")
                    elif done [1]==0 and done[2]==1 :
                        print("第", i + 1, "局:   蓝方胜")
                    else:
                        print("第", i + 1, "局:   平  局")
                    break
                obs = self.run_env(obs)
            print("共", i + 1, "局:   红方胜", battle_results[0], "局:   蓝方胜", battle_results[1], "局:   平局", battle_results[2], "局")
            self._reset()
            print(i + 1, "重置")
        return battle_results

    def run_env(self, obs):
        action = self.get_action(obs)
        return self.step(action)  # 环境推进一步


# 启动单个xsim,同时支持host模式
def main_signal():                                                                      
    from atexit import register
    import os
    address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
    battle_runner = BattleRunnerSignal(config['agents'], address)
    episode_num = config["episode_time"]
    results = battle_runner.run(episode_num)
    # register(lambda: os.system(r"docker stop $(docker ps | grep bvrsim | awk '{print $ 1}')"))  # Failsafe, handles shm leak

    # 输出对抗胜负结果
    print("共", episode_num, "局:   红方胜",results[0],"局:   蓝方胜",results[1],"局:   平局",results[2],"局")


def mult(agents, address):
    battle_runner = BattleRunnerSignal(agents, address, 'port')
    episode_num = config["episode_time"]
    results = battle_runner.run2(episode_num)
    print("共", episode_num, "局:   红方胜",results[0],"局:   蓝方胜",results[1],"局:   平局",results[2],"局")


# 启动多个xsim,不支持实时展示
def main_mult(xsim_num: int):
    po = Pool(POOL_NUM)
    for i in range(xsim_num):
        address = ADDRESS['ip'] + ":" + str(int(ADDRESS['port']) + i)
        po.apply_async(mult, (config['agents'], address))
        print(f"第{i+1}个XSIM启动成功, address=‘{address}’!")

    po.close()
    po.join()



def main():
    """ 只与环境作交互案例，不考虑规则智能体 """
    # 实例化一个xsim环境
    XSIM_env = XSimEnv(80)

    episode_time = config["episode_time"]


    for i in range(episode_time):
        obs = XSIM_env.reset()   # 重置并获取数据
        count = 0
        while True:
            if count > 100:
                break
            print("count:", count)
            # 智能体agent 返回动作
            action = []
            obs = XSIM_env.step(action)   # 环境推进一步
            count += 1


if __name__ == '__main__':
    if ISHOST:
        main_signal()
    else:
        main_mult(XSIM_NUM)

    # main()

