import math
import numpy as np

class EndGame():

    # 额，为啥这个函数这么啰嗦
    def get_done(self, obs):

        # 终止标识， 红方战胜利， 蓝方胜利
        done = [0, 0, 0]
        蓝方战机剩余数量 = len(obs["blue"]["platforminfos"])
        红方战机剩余数量 = len(obs["red"]["platforminfos"])
        # 时间超时，终止
        cur_time = obs["sim_time"]
        if cur_time >= 20 * 60 - 1:
            done[0] = 1
            # 当战损相同时，判断占领中心区域时间
            if 红方战机剩余数量 == 蓝方战机剩余数量:
                if self.red_score > self.blue_score:
                    # print("红方占领中心区域时间更长")
                    done[1] = 1
                elif self.red_score < self.blue_score:
                    # print("蓝方占领中心区域时间更长")
                    done[2] = 1
            # 当战损不同时，判断战损更少一方胜
            else:
                if 红方战机剩余数量 > 蓝方战机剩余数量:
                    # print("红方战损更少")
                    done[1] = 1
                else:
                    # print("蓝方战损更少")
                    done[2] = 1

            # 重置分数
            self.red_score = 0
            self.blue_score = 0
            return done


        # FQX：：的额外添加， 如果在半场时，RL方面，飞机数量为劣势，则直接判定失败
        if cur_time == 15 * 60:
            # 当战损相同时，判断占领中心区域时间
            if 红方战机剩余数量 != 蓝方战机剩余数量:
                红方为RL = (self.player_color == "red")
                蓝方为RL = (self.player_color == "blue")
                if 红方战机剩余数量 > 蓝方战机剩余数量 and 蓝方为RL:
                    # print("[半场] 红方战损更少, RL为蓝方, RL直接失败, ", cur_time)
                    done[0] = 1
                    done[1] = 1
                    self.red_score = 0
                    self.blue_score = 0
                    return done
                elif 红方战机剩余数量 < 蓝方战机剩余数量 and 红方为RL:
                    # print("[半场] 蓝方战损更少, RL为红方, RL直接失败, ", cur_time)
                    done[0] = 1
                    done[2] = 1
                    self.red_score = 0
                    self.blue_score = 0
                    return done
                else:
                    pass
                    # if 红方为RL:
                    #     print('[半场] RL方剩余:', 红方战机剩余数量, '敌方剩余:', 蓝方战机剩余数量)
                    # else:
                    #     print('[半场] RL方剩余:', 蓝方战机剩余数量, '敌方剩余:', 红方战机剩余数量)



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
            # print("红方有人机阵亡")
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
            # print("蓝方有人机阵亡")
            done[0] = 1
            done[1] = 1

        if done[0] == 1:
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
                # print("红方无弹")
                done[0] = 1
                done[2] = 1
            else:
                flag = True
                for red_missile in obs['red']['missileinfos']:
                    if red_missile["Identification"] == "红方":
                        flag = False
                        break
                if flag:
                    # print("红方无弹")
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
                # print("蓝方无弹")
                done[0] = 1
                done[1] = 1
            else:
                flag = True
                for blue_missile in obs['blue']['missileinfos']:
                    if blue_missile["Identification"] == "蓝方":
                        flag = False
                        break
                if flag:
                    # print("蓝方无弹")
                    done[0] = 1
                    done[1] = 1

        if done[0] == 1:
            self.red_score = 0
            self.blue_score = 0
            return done
        return done

