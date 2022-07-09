from ..agent import Agent
from ..env_cmd import CmdEnv
from UTIL.colorful import *
from UTIL.tensor_ops import dir2rad, np_softmax, reg_rad_at, reg_deg_at, reg_rad, repeat_at
import numpy as np
from .tools import distance_matrix
import uuid

'''
    self.coop_list[coop_uuid] = {
        'target_to_hit': op,
        'primary_uav': 主攻,
        'secondary_uav': 副攻,
        'coop_uid': coop_uuid,
        'prior_ms': None,
        'follow_ms': None,
        'valid': True,
    }
'''


class MS_exe():
    # 最底层的导弹发射逻辑
    def ms_launch(self):
        for p in self.my_planes:
            has_not_fire = True
            fired_target_id = None
            for ms_todo in p.ms_to_launch:
                # >> debug
                # for ms in self.ms: assert ms.flying_time==0
                '''
                {
                    'target_ID_to_hit': self.coop_list[req_uuid]['target_to_hit'].ID,
                    'launch_state': None,
                    'related_uuid': req_uuid,
                    'type': 'prior_ms',
                    'req_finished':False
                }
                '''
                assert not ms_todo['req_finished']  # 假定已经完成的请求在上一个循环中已经被清理干净了
                if 'careful_launch' in ms_todo:
                    if ms_todo['careful_launch']:
                        if ms_todo['target_ID_to_hit'] in [op.ID for op in p.in_attack_deadzoo]:
                            ## print亮黄('careful 模式，继续发射')
                            pass
                        else:
                            ms_todo['req_finished'] = True
                            ms_todo['valid'] = False
                            ## print亮黄('careful 模式，放弃发射')

                # ! 清除空目标的发射
                target = self.find_plane_by_id(ms_todo['target_ID_to_hit'])
                if target is None:
                    ms_todo['req_finished'] = True
                    # assert (not self.coop_list[ms_todo['related_uuid']]['valid']) or ()
                    ms_todo['valid'] = False
                    ## print亮绿('☆ 导弹发射委托没完成，但目标已经被摧毁了！')
                # ！以上清除空目标的发射

                # if not has_not_fire: 
                    # assert fired_target_id != ms_todo['target_ID_to_hit'] 例外是有人机，可以被多轮打击
                    ## print('尝试')
                    # pass
                    # continue

                # 把飞机的ms_todo，将launch_state转换为given_cmd
                if ms_todo['launch_state'] is None:
                    ms_todo['launch_state'] = 'given_cmd'
                    has_not_fire = False
                    fired_target_id=ms_todo['target_ID_to_hit']
                    self.cmd_list.append(CmdEnv.make_attackparam(p.ID, ms_todo['target_ID_to_hit'], 1))
                    ## print亮黄('☆ 导弹发射命令发出，发射者:', p.Name, ' 目标:', self.find_plane_by_id(ms_todo['target_ID_to_hit']).Name)
                    continue

                # 是否已经有导弹被确认发射
                assert ms_todo['launch_state'] == 'given_cmd'
                has_ms = [
                    ms for ms in self.ms 
                    if ms.EngageTargetID==ms_todo['target_ID_to_hit'] and
                    ms.LauncherID==p.ID and
                    ms.flying_time==0
                ]   # 筛选出刚发射的导弹
                if len(has_ms)==0: 
                    target = self.find_plane_by_id(ms_todo['target_ID_to_hit'])
                    if target is None:
                        ms_todo['req_finished'] = True
                        # assert (not self.coop_list[ms_todo['related_uuid']]['valid']) or ()
                        ms_todo['valid'] = False
                        ## print亮绿('☆ 导弹发射委托没完成，但目标已经被摧毁了！发射者:', p.Name)
                        continue
                    else:
                        self.cmd_list.append(CmdEnv.make_attackparam(p.ID, ms_todo['target_ID_to_hit'], 1))
                        ## print亮绿('☆ 导弹没有发射，再次发出攻击指令！！需要调试! 发射者:', p.Name,  ' 目标:', self.find_plane_by_id(ms_todo['target_ID_to_hit']).Name)
                        continue
                
                launched_ms = has_ms[0]
                uuid_ = ms_todo['related_uuid']
                if ms_todo['type'] == 'prior_ms':
                    assert self.coop_list[uuid_]['prior_ms'] is None
                    self.coop_list[uuid_].update({
                        'prior_ms':launched_ms
                    })
                elif ms_todo['type'] == 'follow_ms':
                    if uuid_ in self.coop_list:
                        self.coop_list[uuid_].update({
                            'follow_ms':launched_ms
                        })
                elif ms_todo['type'] == 'third_ms':
                    if uuid_ in self.coop_list:
                        self.coop_list[uuid_].update({
                        'third_ms':launched_ms
                        })
                elif ms_todo['type'] == 'solo_ms':
                    if uuid_ in self.coop_list:
                        self.solo_list[uuid_].update({
                            'ms':launched_ms
                        })
                else:
                    assert False
                # 至实际上检测到导弹为止，实际的委托已经完成
                ms_todo['req_finished'] = True
                # 后面将这个ms_todo清除，given_cmd->清除
                ## print亮蓝('☆ 导弹发射委托确认完成！发射者:', p.Name, ' 目标:', self.find_plane_by_id(ms_todo['target_ID_to_hit']).Name)
                

            tmp = list(filter(lambda req: (req['req_finished']), p.ms_to_launch))
            # if len(tmp)>0: ## print亮绿('☆ 清除委托：', tmp)
            # 循环过后，清理已经完成的委托
            p.ms_to_launch = list(filter(lambda req: (not req['req_finished']), p.ms_to_launch))