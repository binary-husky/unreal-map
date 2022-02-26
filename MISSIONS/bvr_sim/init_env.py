import os, sys, math
import numpy as np
from .env.xsim_manager import XSimManager
from .env.communication_service import CommunicationService
from .env.env_runner import EnvRunner
class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with

class ScenarioConfig(object): # ADD_TO_CONF_SYSTEM 加入参数搜索路径 do not remove this comment !!!

    ################ needed by the core of hmp ################
    N_TEAM = 1  
    N_AGENT_EACH_TEAM = [5,]
    AGENT_ID_EACH_TEAM = [range(0,5),]
    TEAM_NAMES = ['ALGORITHM.None->None',] 

    ## If the length of action array == the number of teams, set ActAsUnity to True
    ## If the length of action array == the number of agents, set ActAsUnity to False
    ActAsUnity = False  

    ## If the length of reward array == the number of agents, set RewardAsUnity to False
    ## If the length of reward array == 1, set RewardAsUnity to True
    RewardAsUnity = True

    ## If the length of obs array == the number of agents, set ObsAsUnity to False
    ## If the length of obs array == the number of teams, set ObsAsUnity to True
    ObsAsUnity = True

    ################ needed by env itself ################
    render = False
    max_steps_episode = 1000

    ################ needed by some ALGORITHM ################
    state_provided = False
    avail_act_provided = False
    n_actions = 4

    # n_actions = 6 + map_param_registry[map_]['n_enemies']
    # n_actions_cv = ChainVar(
    #     lambda map_:6 + map_param_registry[map_]['n_enemies'], 
    #     chained_with=['map_']
    # )
    obs_vec_length = 6
    return_mat = False
    block_invalid_action = True # sc2 中，需要始终屏蔽掉不可用的动作



def make_bvr_env(env_id, rank):
    return BASE_ENV_PVE(rank)

class BASE_ENV_PVP(object):
    pass    # raise NotImplementedError
    
planeId2IntList = {
    '红有人机':  0, '蓝有人机':  0,
    '红无人机1': 1, '蓝无人机1': 1,
    '红无人机2': 2, '蓝无人机2': 2,
    '红无人机3': 3, '蓝无人机3': 3,
    '红无人机4': 4, '蓝无人机4': 4,
}
class BASE_ENV_PVE(object):
    def __init__(self, rank):
        # xsim引擎控制器
        self.xsim_manager = XSimManager(time_ratio=100)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)
        self.observation_space = None
        self.action_space = None
        # score of episode
        self.red_score = 0
        self.blue_score = 0
        self.count = 0
        self.scores = {"red": {}, "blue": {}}
        self.player_color = "red"
        self.opp_color = "blue"

    def step(self, action: list) -> dict:
        obs = self.communication_service.step(action)
        done = self.get_done(obs)
        assert obs is not None, ("obs is None")
        converted_obs = self.convert_obs(obs)
        reward = 0
        info = {}
        return converted_obs, reward, done, info

    def reset(self):
        if np.random.rand() < 0.5:
            self.player_color, self.opp_color = ("red", "blue")
        else:
            self.player_color, self.opp_color = ("blue", "red")

        self._end()
        self.communication_service.reset()
        obs = self.communication_service.step([])
        converted_obs = self.convert_obs(obs)
        # info = [obs[self.player_color], obs[self.opp_color]]
        info = obs[self.player_color]
        return converted_obs, info

    def _end(self):
        return self.communication_service.end()

    def render(self):
        raise NotImplementedError

    def __del__(self):
        self.xsim_manager.__del__()



    def convert_obs(self, obs):
        # player_obs, opp_obs = (obs[self.player_color], obs[self.opp_color])
        player_obs, _ = (obs[self.player_color], obs[self.opp_color])

        player_encoded_obs = np.array(size=(20, 14)); p=0
        # planeId2Int = planeId2IntList[['Name']]
        for plane in player_obs['platforminfos']:
            player_encoded_obs[p, 0] = planeId2IntList[plane['Name']]
            player_encoded_obs[p, 1] = plane['Type'] - 1
            player_encoded_obs[p, 2] = plane['Availability']
            player_encoded_obs[p, 3] = plane["X"]
            player_encoded_obs[p, 4] = plane["Y"]
            player_encoded_obs[p, 5] = plane["Alt"]
            player_encoded_obs[p, 6] = plane['Heading']
            player_encoded_obs[p, 7] = plane['Pitch']
            player_encoded_obs[p, 8] = plane['Roll']
            player_encoded_obs[p, 9] = plane['Speed']
            player_encoded_obs[p, 10] = plane['CurTime']
            player_encoded_obs[p, 11] = plane['AccMag']
            player_encoded_obs[p, 12] = plane['NormalG']
            player_encoded_obs[p, 13] = plane['LeftWeapon']
            p+=1

        for plane in player_obs['trackinfos']:
            player_encoded_obs[p, 0]  = planeId2IntList[plane['Name']]
            player_encoded_obs[p, 1]  = plane['Type'] - 1
            player_encoded_obs[p, 2]  = plane['Availability']
            player_encoded_obs[p, 3]  = plane["X"]
            player_encoded_obs[p, 4]  = plane["Y"]
            player_encoded_obs[p, 5]  = plane["Alt"]
            player_encoded_obs[p, 6]  = plane['Heading']
            player_encoded_obs[p, 7]  = plane['Pitch']
            player_encoded_obs[p, 8]  = plane['Roll']
            player_encoded_obs[p, 9]  = plane['Speed']
            player_encoded_obs[p, 10] = plane['CurTime']
            player_encoded_obs[p, 11] = 0
            player_encoded_obs[p, 12] = 0
            player_encoded_obs[p, 13] = -1
            p+=1

        
        return player_encoded_obs

    # 额，为啥这个函数这么啰嗦
    def get_done(self, obs):
        self.red_score
        self.blue_score
        self.count
        self.scores
        # print(obs)
        done = [0, 0, 0]  # 终止标识， 红方战胜利， 蓝方胜利

        # 时间超时，终止
        cur_time = obs["sim_time"]
        # print("get_done cur_time:", cur_time)
        if cur_time >= 20 * 60 - 1:
            done[0] = 1
            # 当战损相同时，判断占领中心区域时间
            if len(obs["red"]["platforminfos"]) == len(obs["blue"]["platforminfos"]):
                if self.red_score > self.blue_score:
                    print("红方占领中心区域时间更长")
                    done[1] = 1
                elif self.red_score < self.blue_score:
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
            self.count = self.count + 1
            # 记录分数
            self.scores["red"]["Game" + str(self.count)] = self.red_score
            self.scores["blue"]["Game" + str(self.count)] = self.blue_score
            # 重置分数
            self.red_score = 0
            self.blue_score = 0
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
                    self.red_score = self.red_score + 1
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
                    self.blue_score = self.blue_score + 1
                    # print("Blue Score:", blue_score)
                break
        # print("get_done has_blue_combat:", has_blue_combat)
        if not has_blue_combat:
            print("蓝方有人机阵亡")
            done[0] = 1
            done[1] = 1

        if done[0] == 1:
            self.count = self.count + 1
            self.scores["red"]["Game" + str(self.count)] = self.red_score
            self.scores["blue"]["Game" + str(self.count)] = self.blue_score
            self.red_score = 0
            self.blue_score = 0
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
            self.count = self.count + 1
            self.scores["red"]["Game" + str(self.count)] = self.red_score
            self.scores["blue"]["Game" + str(self.count)] = self.blue_score
            self.red_score = 0
            self.blue_score = 0
            return done
        return done





class Template_Env():
    def __init__(self) -> None:
        self.observation_space = None
        self.action_space = None
        raise NotImplementedError

    def step(self, act):
        # obs: a Tensor with shape (n_agent, ...)
        # reward: a Tensor with shape (n_agent, 1) or (n_team, 1) or (1,)
        # done: a Bool
        # info: a dict
        raise NotImplementedError
        return (ob, reward, done, info)

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # done: a Bool
        raise NotImplementedError
        return ob, info
