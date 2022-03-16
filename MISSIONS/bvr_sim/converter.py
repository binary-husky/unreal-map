import numpy as np
from .agent.env_cmd import CmdEnv

class Converter():

    planeId2IntList = {
        '红有人机':  0, '蓝有人机':  0,
        '红无人机1': 1, '蓝无人机1': 1,
        '红无人机2': 2, '蓝无人机2': 2,
        '红无人机3': 3, '蓝无人机3': 3,
        '红无人机4': 4, '蓝无人机4': 4,
    }

    def convert_obs(self, obs):
        # player_obs, opp_obs = (obs[self.player_color], obs[self.opp_color])
        player_obs, _ = (obs[self.player_color], obs[self.opp_color])

        player_encoded_obs = np.zeros(shape=(20, 14)); p=0
        # planeId2Int = planeId2IntList[['Name']]
        for plane in player_obs['platforminfos']:
            player_encoded_obs[p, 0] = self.planeId2IntList[plane['Name']]
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
            player_encoded_obs[p, 0]  = self.planeId2IntList[plane['Name']]
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
                pass
            
            # if AT!=0:   # Do nothing
            cmd_buffer = self.parse_act_final_HT(cmd_buffer, p, HT)
            cmd_buffer = self.parse_act_final_SP(cmd_buffer, p, SP)
        return cmd_buffer

    def parse_act_case_fire(self, cmd_buffer, p, TT):
        target = self.tran_target(TT)
        if (self.player_color=='red' and p.LeftWeapon<=1) or (self.player_color=='blue' and p.OpLeftWeapon<=1):
            print('saving ms, do not fire')
            return cmd_buffer


        cmd_buffer.append(CmdEnv.make_attackparam(p.ID, target.ID, 1))
        return cmd_buffer


    def parse_act_final_HT(self, cmd_buffer, p, HT):
        p.PlayerHeightSetting = p.MinHeight + (p.MaxHeight - p.MinHeight)*( HT/(self.ScenarioConfig.HeightLevels-1) )
        return cmd_buffer

    def parse_act_final_SP(self, cmd_buffer, p, SP):
        cmd_speed = p.MinSpeed + (p.MaxSpeed - p.MinSpeed)*( SP/(self.ScenarioConfig.SpeedLevels-1) ) # , i = [0,1,2,3,4]
        # 指令速度:1 / 指令加速度:2 / 指令速度和加速度:3 / 指令过载:4 / 指令速度和过载:5 / 指令加速度和过载:6 / 指令速度和加速度和过载:7
        cmd_buffer.append(CmdEnv.make_motioncmdparam(p.ID, 1, cmd_speed, p.MaxAcc, p.MaxOverload))
        return cmd_buffer

    def parse_act_case_track(self, cmd_buffer, p, TT, rad):
        target = self.tran_target(TT)
        delta_to_TT =  target.pos2d - p.pos2d # 向量的方向指向目标
        unit_delta = np.matmul(
            delta_to_TT,
            np.array([[np.cos(rad), np.sin(rad)],
                     [-np.sin(rad), np.cos(rad)]])
        )
        
        H2 = unit_delta[:2] + p.pos2d
        goto_location = [{
            "X": H2[0],
            "Y": H2[1],
            "Z": target.Z   # p.PlayerHeightSetting
        }]
        cmd_buffer.append(self.observer.check_and_make_linepatrolparam(
            p.ID,
            goto_location,
            p.Speed,
            p.MaxAcc,
            p.MaxOverload
        ))
        return cmd_buffer
