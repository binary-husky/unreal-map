#-*-coding:utf-8-*-
"""
@FileName：observation.py
@Description：
@Author：qiwenhao
@Time：2021/5/12 20:11
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
@Project：601
"""
_OBSINIT = None


class ObservationProcessor(object):

    # 解析数据包
    @staticmethod
    def get_obs(Data):
        """
        解析状态信息并组包
        :param Data: bvrsim 返回的态势信息
        :return:　解析后的 bvrsim 态势信息，格式为dict
        """
        
        # 初始化obs数据结构
        if Data is None:
            print("从引擎中获取的数据为空！")
            return None
        obs = {side: dict(platforminfos=[], trackinfos=[], missileinfos=[]) for side in ['blue', 'red']}
        obs["sim_time"] = Data.CurTime  # 仿真时间
        obs["xsim_tag"] = Data.XSimTag  # 引擎标识
        data = Data.IdenInfos
        try:
            for info in data:
                # 己方作战平台信息解析
                for platforminfo in info.PlatformInfos:
                    if platforminfo.Identification == "红方":
                        pl_side = "red"
                    elif platforminfo.Identification == "蓝方":
                        pl_side = "blue"
                    else:
                        continue

                    obs[pl_side]['platforminfos'].append(
                        dict(
                            Name=platforminfo.Name,						# 飞机的名称
                            Identification=platforminfo.Identification,	# 飞机的标识符（表示飞机是红方还是蓝方）
                            ID=platforminfo.ID,							# 飞机的ID（表示飞机的唯一编号）
                            Type=platforminfo.Type,						# 飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
                            Availability=platforminfo.Availability,		# 飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
                            X=platforminfo.X,							# 飞机的当前X坐标（表示飞机的X坐标）
                            Y=platforminfo.Y,							# 飞机的当前Y坐标（表示飞机的Y坐标）
                            Lon=platforminfo.Lon,						# 飞机的当前所在经度（表示飞机的所在经度）
                            Lat=platforminfo.Lat,						# 飞机的当前所在纬度（表示飞机的所在纬度）
                            Alt=platforminfo.Alt,						# 飞机的当前所在高度（表示飞机的所在高度）
                            Heading=platforminfo.Heading,				# 飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
                            Pitch=platforminfo.Pitch,					# 飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
                            Roll=platforminfo.Roll,						# 飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
                            Speed=platforminfo.Speed,					# 飞机的当前速度（飞机的当前速度）
                            CurTime=platforminfo.CurTime,				# 当前时间（当前时间）
                            AccMag=platforminfo.AccMag,					# 飞机的指令加速度（飞机的指令加速度）
                            NormalG=platforminfo.NormalG,				# 飞机的指令过载（飞机的指令过载）
                            IsLocked=platforminfo.IsLocked,				# 飞机是否被敌方导弹锁定（飞机是否被敌方导弹锁定）
                            Status=platforminfo.Status,					# 飞机的当前状态（飞机的当前状态）	
                            LeftWeapon=platforminfo.LeftWeapon			# 飞机的当前剩余导弹数（飞机的当前剩余导弹数）	
                        )
                    )

                # 目标信息解析
                for trackinfo in info.TargetInfos:
                    if trackinfo.Identification == "蓝方":
                        ti_side = "red"
                    elif trackinfo.Identification == "红方":
                        ti_side = "blue"
                    else:
                        continue
                    obs[ti_side]['trackinfos'].append(
                        dict(
                            Name=trackinfo.Name,						# 敌方飞机的名称
                            Identification=trackinfo.Identification,	# 敌方飞机的标识符（表示敌方飞机是红方还是蓝方）
                            ID=trackinfo.ID,							# 敌方飞机的ID（表示飞机的唯一编号）
                            Type=trackinfo.Type,						# 敌方飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
                            Availability=trackinfo.Availability,		# 敌方飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
                            X=trackinfo.X,								# 敌方飞机的当前X坐标（表示飞机的X坐标）
                            Y=trackinfo.Y,								# 敌方飞机的当前Y坐标（表示飞机的Y坐标）
                            Lon=trackinfo.Lon,							# 敌方飞机的当前所在经度（表示飞机的所在经度）
                            Lat=trackinfo.Lat,							# 敌方飞机的当前所在纬度（表示飞机的所在纬度）
                            Alt=trackinfo.Alt,							# 敌方飞机的当前所在高度（表示飞机的所在高度）
                            Heading=trackinfo.Heading,					# 敌方飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
                            Pitch=trackinfo.Pitch,						# 敌方飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
                            Roll=trackinfo.Roll,						# 敌方飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
                            Speed=trackinfo.Speed,						# 敌方飞机的当前速度（飞机的当前速度）
                            CurTime=trackinfo.CurTime,					# 当前时间（当前时间）
                            IsLocked=trackinfo.IsLocked					# 敌方飞机是否被敌方导弹锁定（飞机是否被己方导弹锁定）
                        )
                    )

                # 来袭导弹信息解析
                for missileinfo in info.MissileInfos:
                    if missileinfo.Identification == "红方":
                        mi_side = "blue"
                    elif missileinfo.Identification == "蓝方":
                        mi_side = "red"
                    else:
                        continue
                    obs[mi_side]['missileinfos'].append(
                        dict(
                            Name=missileinfo.Name,						# 敌方导弹的名称
                            Identification=missileinfo.Identification,	# 敌方导弹的标识符（表示敌方导弹是红方还是蓝方）
                            ID=missileinfo.ID,							# 敌方导弹的ID（表示导弹的唯一编号）
                            Type=missileinfo.Type,                      # 敌方导弹的类型（表示导弹的类型，其中导弹类型为 3）
                            Availability=missileinfo.Availability,		# 敌方导弹的可用性（表示导弹的可用性，范围为0到1,为1表示飞机存活，0表示导弹已爆炸）
                            X=missileinfo.X,							# 敌方导弹的当前X坐标（表示导弹的X坐标）
                            Y=missileinfo.Y,							# 敌方导弹的当前Y坐标（表示导弹的Y坐标）
                            Lon=missileinfo.Lon,						# 敌方导弹的当前所在经度（表示导弹的所在经度）
                            Lat=missileinfo.Lat,						# 敌方导弹的当前所在纬度（表示导弹的所在纬度）
                            Alt=missileinfo.Alt,						# 敌方导弹的当前所在高度（表示导弹的所在高度）
                            Heading=missileinfo.Heading,				# 敌方导弹的当前朝向角度（导弹的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
                            Pitch=missileinfo.Pitch,					# 敌方导弹的当前俯仰角度（导弹的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
                            Roll=missileinfo.Roll,						# 敌方导弹的当前滚转角度（导弹的当前滚转,范围为-180°到180° ）
                            Speed=missileinfo.Speed,					# 敌方导弹的当前速度（导弹的当前速度）
                            CurTime=missileinfo.CurTime,				# 当前时间（当前时间）
                            LauncherID=missileinfo.LauncherID,          # 敌方导弹的发射者ID（敌方导弹的发射者ID）
                            EngageTargetID=missileinfo.EngageTargetID	# 敌方已发射导弹攻击目标的ID（我方已发射导弹攻击目标的ID）
                        )
                    )

                # 友方导弹信息解析
                for missileinfo in info.MissileInfos:
                    if missileinfo.Identification == "红方":
                        mi_side = "red"
                    elif missileinfo.Identification == "蓝方":
                        mi_side = "blue"
                    else:
                        continue
                    obs[mi_side]['missileinfos'].append(
                        dict(
                            Name=missileinfo.Name,						# 我方已发射导弹的名称
                            Identification=missileinfo.Identification,	# 我方已发射导弹的标识符（表示我方已发射导弹是红方还是蓝方）
                            ID=missileinfo.ID,							# 我方已发射导弹的ID（表示导弹的唯一编号）
                            Type=missileinfo.Type,						# 我方已发射导弹的类型（表示导弹的类型，其中导弹类型为 3）
                            Availability=missileinfo.Availability,		# 我方已发射导弹的可用性（表示导弹的可用性，范围为0到1,为1表示飞机存活，0表示导弹已爆炸）
                            X=missileinfo.X,							# 我方已发射导弹的当前X坐标（表示导弹的X坐标）
                            Y=missileinfo.Y,							# 我方已发射导弹的当前Y坐标（表示导弹的Y坐标）
                            Lon=missileinfo.Lon,						# 我方已发射导弹的当前所在经度（表示导弹的所在经度）
                            Lat=missileinfo.Lat,						# 我方已发射导弹的当前所在纬度（表示导弹的所在纬度）
                            Alt=missileinfo.Alt,						# 我方已发射导弹的当前所在高度（表示导弹的所在高度）
                            Heading=missileinfo.Heading,				# 我方已发射导弹的当前朝向角度（导弹的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
                            Pitch=missileinfo.Pitch,					# 我方已发射导弹的当前俯仰角度（导弹的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
                            Roll=missileinfo.Roll,						# 我方已发射导弹的当前滚转角度（导弹的当前滚转,范围为-180°到180°）
                            Speed=missileinfo.Speed,					# 我方已发射导弹的当前速度（导弹的当前速度）
                            CurTime=missileinfo.CurTime,				# 当前时间（当前时间）
                            LauncherID=missileinfo.LauncherID,			# 我方已发射导弹的发射者ID（我方已发射导弹的发射者ID）
                            EngageTargetID=missileinfo.EngageTargetID	# 我方已发射导弹攻击目标的ID（我方已发射导弹攻击目标的ID）
                        )
                    )


            global _OBSINIT

            if _OBSINIT is None:
                _OBSINIT = obs
        except Exception as e:
            print("解析数据异常～")
        return obs
