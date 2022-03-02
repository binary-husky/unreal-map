import os, sys, math
import numpy as np
from .env.xsim_manager import XSimManager
from .env.communication_service import CommunicationService
from .env.env_runner import EnvRunner
from .agent.env_cmd import CmdEnv
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
        # opponent AI
        from .agent.yiteam_final_commit_v7.Yi_team import Yi_team
        self.OPP_CONTROLLER_CLASS = Yi_team

        # an observer keeps track of agent info such as distence
        from .agent.dummy_observer.base import Baseclass
        self.OBSERVER_CLASS = Baseclass

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
        self.just_suffer_reset = True
        self.internal_step = 4

        # self.n_agents
        self.n_agents = 5
        self.n_opp = 5
        
        # the id of env among parallel envs, only rank=0 thread is allowed to render
        self.rank = rank


    def step(self, raw_action):
        # convert raw_action Tensor from RL to actions recognized by the game
        player_action = self.parse_raw_action(raw_action)
        
        for _ in range(self.internal_step):
            # move the opponents
            opp_action = self.opp_controller.step(self.raw_obs["sim_time"], self.raw_obs[self.opp_color])

            # commmit actions to the environment
            action = player_action + opp_action
            self.raw_obs = self.communication_service.step(action)
            player_action = []

            # an observer keeps track of agent info such as distence
            self.observer.observe(self.raw_obs["sim_time"], self.raw_obs['red'])

            done, red_win, blue_win = self.get_done(self.raw_obs)
            done = (done>0); 
            win = {'red':(red_win>0), 'blue':(blue_win>0), 'done':done}
            if done: 
                break
        
        # use observer's info to render
        if ScenarioConfig.render and self.rank==0: self.advanced_render(self.raw_obs, win)

        # system get done
        assert self.raw_obs is not None, ("obs is None")
        converted_obs = self.convert_obs(self.raw_obs)
        reward = 0
        info = {'win':win[self.player_color]}
        return converted_obs, reward, done, info

    def reset(self):
        self.just_suffer_reset = True
        # an observer keeps track of agent info such as distence
        if np.random.rand() < 0.5:
            self.player_color, self.opp_color = ("red", "blue")
            self.opp_controller = self.OPP_CONTROLLER_CLASS('blue', {"side": 'blue'})
        else:
            self.player_color, self.opp_color = ("blue", "red")
            self.opp_controller = self.OPP_CONTROLLER_CLASS('red', {"side": 'red'})

        self._end()
        self.communication_service.reset()
        self.raw_obs = self.communication_service.step([])
        # observer's first info feedin
        self.observer = self.OBSERVER_CLASS('red', {"side": 'red'})
        self.observer.observe(self.raw_obs["sim_time"], self.raw_obs['red'])

        converted_obs = self.convert_obs(self.raw_obs)
        # info = [obs[self.player_color], obs[self.opp_color]]
        info = self.raw_obs[self.player_color]
        return converted_obs, info

    def advanced_render(self, raw_obs, win):
        if not hasattr(self, '可视化桥'):
            from VISUALIZE.mcom import mcom
            self.可视化桥 = mcom(path='RECYCLE/v2d_logger/', draw_mode='Threejs')
            self.可视化桥.初始化3D()
            # self.可视化桥.设置样式('background', color='White') # 注意不可以省略参数键值'color=' ！！！
            # self.可视化桥.设置样式('sky')   # adjust parameters https://threejs.org/examples/?q=sky#webgl_shaders_sky
            # self.可视化桥.设置样式('skybox6side', 
            #     posx='/wget/mars_textures/mars_posx.jpg',
            #     negx='/wget/mars_textures/mars_negx.jpg',
            #     posy='/wget/mars_textures/mars_posy.jpg',
            #     negy='/wget/mars_textures/mars_negy.jpg',
            #     posz='/wget/mars_textures/mars_posz.jpg',
            #     negz='/wget/mars_textures/mars_negz.jpg',
            # )


            self.可视化桥.设置样式('skybox6side', 
                posx='/wget/snow_textures/posx.jpg',
                negx='/wget/snow_textures/negx.jpg',
                posy='/wget/snow_textures/negy.jpg',
                negy='/wget/snow_textures/posy.jpg',
                posz='/wget/snow_textures/posz.jpg',
                negz='/wget/snow_textures/negz.jpg',
            )


            # self.可视化桥.设置样式('font', fontPath='/examples/fonts/ttf/HGXH_CNKI.TTF', fontLineHeight=1000) # 注意不可以省略参数键值'font_path=' ！！！
            self.可视化桥.设置样式('font', fontPath='/examples/fonts/ttf/FZYTK.TTF', fontLineHeight=1500) # 注意不可以省略参数键值'font_path=' ！！！
            self.可视化桥.其他几何体之旋转缩放和平移('BOX', 'BoxGeometry(1,1,1)',   0,0,0,  1,1,1, 0,0,0) 
            self.可视化桥.其他几何体之旋转缩放和平移('OCT', 'OctahedronGeometry(1,0)', 0,0,0,  1,1,1, 0,0,0)   # 八面体
            self.可视化桥.其他几何体之旋转缩放和平移('Plane', 'fbx=/examples/files/plane.fbx', -np.pi/2, 0, np.pi/2,  1,1,1, 0,0,0)   # 八面体
            self.可视化桥.recorded_objs = {'red':[], 'blue':[], 'ms':[]}
            self.可视化桥.上次结果 = ''

        if self.just_suffer_reset or raw_obs["sim_time"]==4:
            self.just_suffer_reset = False
            for _ in range(10): 
                if win[self.player_color]:
                    self.可视化桥.上次结果 = ' 本局RL胜利\n本局RL为%s队'%self.player_color
                else:
                    self.可视化桥.上次结果 = ' 本局RL战败\n本局RL为%s队'%self.player_color
                self.可视化桥.发送几何体(
                    'BOX|%d|%s|10'%(-1, 'White'),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                    0, 60, 0,                # 三维位置，3/6dof
                    ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
                    opacity=0,              # 透明度，1为不透明
                    label=self.可视化桥.上次结果,               # 显示标签，空白不显示，用'\n'换行
                    label_color='Yellow',    # 标签颜色
                    # label_offset=np.array([0,2,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                )
                self.可视化桥.空指令()
                self.可视化桥.结束关键帧()
            self.可视化桥.set_env('clear_everything')
            self.可视化桥.结束关键帧()

        # 这里走了个捷径
        red_planes = self.observer.my_planes
        blue_planes = self.observer.op_planes
        missiles = self.observer.ms
        z_scale = 5
        for p in red_planes:
            color_ = 'Red' if p.alive else 'Black'
            size = 3 if '有人机' in p.Name else 1.5
            pitch_ = -np.arctan(np.tan(p.Pitch)*z_scale)
            self.可视化桥.发送几何体(
                'Plane|%d|%s|%.3f'%(p.ID, color_, size),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                p.X/1e3, p.Y/1e3, p.Z/1e3*z_scale,                # 三维位置，3/6dof
                ro_x=p.Roll, ro_y=pitch_, ro_z=-p.Heading+np.pi/2, ro_order='ZYX',# 欧拉旋转变换，3/6dof
                opacity=1,              # 透明度，1为不透明
                label=''.join([p.Name,'\n剩余载弹 %d'%p.LeftWeapon]),               # 显示标签，空白不显示，用'\n'换行
                label_color='Yellow',    # 标签颜色
                label_offset=np.array([0,4,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                track_n_frame=60,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
                track_tension=0,      # 轨迹曲线的平滑度，0为不平滑，推荐设置0不平滑
                track_color='Red',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
            )
        # label=''.join([p.Name,'\n剩余载弹 %d'%p.OpLeftWeapon,'\n速度 %d'%int(p.Speed)]),               # 显示标签，空白不显示，用'\n'换行
        # label_offset=np.array([0,2,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
        for p in blue_planes:
            color_ = 'Blue' if p.alive else 'Black'
            size = 3 if '有人机' in p.Name else 1.5
            pitch_ = -np.arctan(np.tan(p.Pitch)*z_scale)
            self.可视化桥.发送几何体(
                'Plane|%d|%s|%.3f'%(p.ID, color_, size),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                p.X/1e3, p.Y/1e3, p.Z/1e3*z_scale,                # 三维位置，3/6dof
                ro_x=p.Roll, ro_y=pitch_, ro_z=-p.Heading+np.pi/2, ro_order='ZYX',# 欧拉旋转变换，3/6dof
                opacity=1,              # 透明度，1为不透明
                label=''.join([p.Name,'\n剩余载弹 %d'%p.OpLeftWeapon]),               # 显示标签，空白不显示，用'\n'换行
                label_offset=np.array([0,4,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                label_color='Yellow',    # 标签颜色
                track_n_frame=60,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
                track_tension=0,      # 轨迹曲线的平滑度，0为不平滑，推荐设置0不平滑
                track_color='Blue',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
            )            
        for p in missiles:
            color_ = 'Pink' if p.Identification=='红方' else 'BlueViolet'
            color_ = color_ if p.alive else 'Black'
            pitch_ = -np.arctan(np.tan(p.Pitch)*z_scale)
            self.可视化桥.发送几何体(
                'BOX|%d|%s|0.5'%(p.ID, color_),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
                p.X/1e3, p.Y/1e3, p.Z/1e3*z_scale,                # 三维位置，3/6dof
                ro_x=p.Roll, ro_y=pitch_, ro_z=-p.Heading+np.pi/2, ro_order='ZYX',# 欧拉旋转变换，3/6dof
                opacity=1,              # 透明度，1为不透明
                label='飞行时间%d'%p.flying_time if p.alive else '',               # 显示标签，空白不显示，用'\n'换行
                label_color='Yellow',    # 标签颜色
                # label_offset=np.array([0,2,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
                track_n_frame=120,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
                track_tension=0,      # 轨迹曲线的平滑度，0为不平滑，推荐设置0不平滑
                track_color=color_,    # 轨迹的颜色显示，输入js颜色名或者hex值均可
            )


        if win['done']:
            if win[self.player_color]:
                self.可视化桥.上次结果 = ' 上局RL胜利\n本局RL为%s队'%self.player_color
            else:
                self.可视化桥.上次结果 = ' 上局RL战败\n本局RL为%s队'%self.player_color

        self.可视化桥.发送几何体(
            'BOX|%d|%s|10'%(-1, 'White'),  # 填入核心参量： “已声明的形状|几何体的唯一ID标识|颜色|整体大小”
            0, 60, 0,                # 三维位置，3/6dof
            ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
            opacity=0,              # 透明度，1为不透明
            label=self.可视化桥.上次结果,               # 显示标签，空白不显示，用'\n'换行
            label_color='Yellow',    # 标签颜色
            # label_offset=np.array([0,2,2]), # 标签与物体之间的相对位置，实验选项，不建议手动指定
        )


        self.可视化桥.结束关键帧()

        return

    def _end(self):
        return self.communication_service.end()

    def __del__(self):
        print('[init_env.py] bvr class del')
        self.xsim_manager.__del__()

    def parse_raw_action(self, raw_action):
        # parse raw action into cmd_buffer
        cmd_buffer = []
        assert raw_action.shape[0] == self.n_agents
        assert raw_action.shape[1] == 4
        assert self.n_opp == self.n_agents

        if self.player_color=='red':
            player_planes, opp_planes = (self.observer.my_planes, self.observer.op_planes)
        else:
            player_planes, opp_planes = (self.observer.op_planes, self.observer.my_planes)




        for index, p in enumerate(player_planes):
            # Assert the order of planes are correct
            if index==0: assert '有人机' in p.Name
            else: assert str(index) in p.Name
            # 
            AT = raw_action[index, 0]   # action type
            TT = raw_action[index, 1]   # target type (opp + teammate + incoming ms)
            HT = raw_action[index, 2]   # height sel
            SP = raw_action[index, 3]   # speed sel
            # switch case for AT
            if AT==0:   # Do nothing
                pass
            elif AT==1: # parse_act_case_track
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=0)
            elif AT==2: # parse_act_case_reverseTrack
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=np.pi)
            elif AT==3: # parse_act_case_3clockTrack
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=-np.pi/2)
            elif AT==4: # parse_act_case_9clockTrack
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=+np.pi/2)
            elif AT==5:
                cmd_buffer = self.parse_act_case_fire(cmd_buffer, p, TT)
            
            # if AT!=0:   # Do nothing
            # cmd_buffer = self.parse_act_final_HT(cmd_buffer, HT)
            # cmd_buffer = self.parse_act_final_SP(cmd_buffer, SP)
        return cmd_buffer

    def parse_act_case_fire(self, cmd_buffer, p, TT):
        target = self.tran_target(TT)
        cmd_buffer.append(CmdEnv.make_attackparam(p.ID, target.ID, 1))
        return cmd_buffer

    def tran_target(self, TT):
        Trans = {
            "red":  ['蓝有人机','蓝无人机1','蓝无人机2','蓝无人机3','蓝无人机4', '红有人机','红无人机1','红无人机2','红无人机3','红无人机4'],
            "blue": ['红有人机','红无人机1','红无人机2','红无人机3','红无人机4', '蓝有人机','蓝无人机1','蓝无人机2','蓝无人机3','蓝无人机4',],
        }
        if TT < 10:
            target_name = Trans[self.player_color][int(TT)]
        else:
            assert False, ('here is the missile target')
        target = self.observer.find_plane_by_name(target_name)
        return target

    def parse_act_case_track(self, cmd_buffer, p, TT, rad):
        target = self.tran_target(TT)
        delta_to_TT =  target.pos2d - p.pos2d # 向量的方向指向目标
        unit_delta = np.matmul(
            delta_to_TT,
            np.array([[np.cos(rad), np.sin(rad)],
                  [np.sin(-rad), np.cos(rad)]]))
        
        H2 = unit_delta[:2] * 100e3 + p.pos2d
        goto_location = [{
            "X": H2[0],
            "Y": H2[1],
            "Z": p.Z
        }]
        cmd_buffer.append(self.observer.check_and_make_linepatrolparam(
            p.ID,
            goto_location,
            p.Speed,
            p.MaxAcc,
            p.MaxOverload
        ))
        return cmd_buffer


    def convert_obs(self, obs):
        # player_obs, opp_obs = (obs[self.player_color], obs[self.opp_color])
        player_obs, _ = (obs[self.player_color], obs[self.opp_color])

        player_encoded_obs = np.zeros(shape=(20, 14)); p=0
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
