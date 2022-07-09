"""
@FileName：env_runner.py.py
@Description：
@Author：liyelei
@Time：2021/5/20 18:08
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from ..env.xsim_env import XSimEnv
import math, time
import math

red_score = 0
blue_score = 0
count = 0
scores = {"red": {}, "blue": {}}
class EnvRunner(XSimEnv):

    def __init__(self, agents, address, mode: str = 'host'):
        """
        对战环境初始化
        @param agents: 智能体列表
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        # self.end()
        # print('clear env now!')
        # time.sleep(3)
        print("初始化 EnvRunner")
        XSimEnv.__init__(self, TimeRatio, address, IMAGE, mode)
        self.agents = {}
        self.__init_agents(agents)

    def __init_agents(self, agents):
        """
        根据配置信息构建红蓝双方智能体
        @param agents: 智能体列表
        @return:
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        red_cls = agents['red']
        blue_cls = agents['blue']
        red_agent = red_cls('red', {"side": 'red'})
        self.agents["red"] = red_agent
        blue_agent = blue_cls('blue', {"side": 'blue'})
        self.agents["blue"] = blue_agent

    def _end(self):

        self.end()

    def _reset(self):
        """
        智能体、环境重置
        @return:
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """

        # 智能体重置
        for side, agent in self.agents.items():
            agent.reset()

        # 环境重置
        self.reset()
        obs = self.step([])
        # while obs["sim_time"] > 10:
        #     obs = self.step([])
        return obs

    def get_action(self, obs):
        """
        从智能体中获取动作
        @param obs: 状态信息
        @return: 动作指令
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        actions = []
        cur_time = obs["sim_time"]
        for side, agent in self.agents.items():
            cmd_list = self._agent_step(agent, cur_time, obs[side])
            # print(cmd_list)
            actions.extend(cmd_list)

        return actions

    def _agent_step(self, agent, cur_time, obs_side):
        """
        获取一方智能体动作指令
        @param agent:
        @param cur_time:
        @param obs_side:
        @return:
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        cmd_list = agent.step(cur_time, obs_side)
        return cmd_list

    def get_done(self, obs):
        """
        推演是否结束
        @param obs: 环境状态信息
        @return: done列表信息
        """
        global red_score
        global blue_score
        global count
        global scores
        # print(obs)
        done = [0, 0, 0]  # 终止标识， 红方战胜利， 蓝方胜利

        # 时间超时，终止
        cur_time = obs["sim_time"]
        # print("get_done cur_time:", cur_time)
        if cur_time >= 20 * 60 - 1:
            done[0] = 1
            # 当战损相同时，判断占领中心区域时间
            if len(obs["red"]["platforminfos"]) == len(obs["blue"]["platforminfos"]):
                if red_score > blue_score:
                    print("红方占领中心区域时间更长")
                    done[1] = 1
                elif red_score < blue_score:
                    print("蓝方占领中心区域时间更长")
                    done[2] = 1
            # 当战损不同时，判断战损更少一方胜
            else:
                if len(obs["red"]["platforminfos"]) > len(obs["blue"]["platforminfos"]):
                    print("红方战损更少")
                    done[1] = 1
                else:
                    print("蓝方战损更少")
                    done[2] = 1
            # 计算次数
            count = count + 1
            # 记录分数
            scores["red"]["Game" + str(count)] = red_score
            scores["blue"]["Game" + str(count)] = blue_score
            # 重置分数
            red_score = 0
            blue_score = 0
            return done

        # 红方有人机全部战损就终止
        red_obs_units = obs["red"]["platforminfos"]
        has_red_combat = False
        for red_obs_unit in red_obs_units:
            red_obs_unit_name = red_obs_unit["Name"]
            if red_obs_unit_name.split("_")[0] == "红有人机":
                has_red_combat = True
                # 判断红方有人机是否在中心区域
                distance_to_center = math.sqrt(
                    red_obs_unit["X"] * red_obs_unit["X"]
                    + red_obs_unit["Y"] * red_obs_unit["Y"]
                    + (red_obs_unit["Alt"] - 9000) * (red_obs_unit["Alt"] - 9000))
                # print("Red distance:", distance_to_center)
                if distance_to_center <= 50000 and red_obs_unit["Alt"] >= 2000 and red_obs_unit['Alt'] <= 16000:
                    red_score = red_score + 1
                    # print("Red Score:", red_score)
                break
        if not has_red_combat:
            print("红方有人机阵亡")
            done[0] = 1
            done[2] = 1

        # 蓝方有人机全部战损就终止
        blue_obs_units = obs["blue"]["platforminfos"]
        has_blue_combat = False
        for blue_obs_unit in blue_obs_units:
            blue_obs_unit_name = blue_obs_unit["Name"]
            if blue_obs_unit_name.split("_")[0] == "蓝有人机":
                has_blue_combat = True
                # 判断蓝方有人机是否在中心区域
                distance_to_center = math.sqrt(
                    blue_obs_unit["X"] * blue_obs_unit["X"]
                    + blue_obs_unit["Y"] * blue_obs_unit["Y"]
                    + (blue_obs_unit["Alt"] - 9000) * (blue_obs_unit["Alt"] - 9000))
                # print("Blue distance:", distance_to_center)
                if distance_to_center <= 50000 and blue_obs_unit["Alt"] >= 2000 and blue_obs_unit['Alt'] <= 16000:
                    blue_score = blue_score + 1
                    # print("Blue Score:", blue_score)
                break
        # print("get_done has_blue_combat:", has_blue_combat)
        if not has_blue_combat:
            print("蓝方有人机阵亡")
            done[0] = 1
            done[1] = 1

        if done[0] == 1:
            count = count + 1
            scores["red"]["Game" + str(count)] = red_score
            scores["blue"]["Game" + str(count)] = blue_score
            red_score = 0
            blue_score = 0
            return done

        # 红方没有导弹就终止
        has_red_missile = False
        for red_obs_unit in red_obs_units:
            if red_obs_unit["LeftWeapon"] > 0:
                has_red_missile = True
                break
        if not has_red_missile:
            if len(obs["red"]["missileinfos"]) == 0:
                print("红方无弹")
                done[0] = 1
                done[2] = 1
            else:
                flag = True
                for red_missile in obs['red']['missileinfos']:
                    if red_missile["Identification"] == "红方":
                        flag = False
                        break
                if flag:
                    print("红方无弹")
                    done[0] = 1
                    done[2] = 1

        # 蓝方没有导弹就终止
        has_blue_missile = False
        for blue_obs_unit in blue_obs_units:
            if blue_obs_unit["LeftWeapon"] > 0:
                has_blue_missile = True
                break
        if not has_blue_missile:
            if len(obs["blue"]["missileinfos"]) == 0:
                print("蓝方无弹")
                done[0] = 1
                done[1] = 1
            else:
                flag = True
                for blue_missile in obs['blue']['missileinfos']:
                    if blue_missile["Identification"] == "蓝方":
                        flag = False
                        break
                if flag:
                    print("蓝方无弹")
                    done[0] = 1
                    done[1] = 1

        if done[0] == 1:
            count = count + 1
            scores["red"]["Game" + str(count)] = red_score
            scores["blue"]["Game" + str(count)] = blue_score
            red_score = 0
            blue_score = 0
            return done
        return done



