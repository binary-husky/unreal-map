#-*-coding:utf-8-*-
"""
@FileName：xsim_env.py
@Description：xsim环境交互类
@Author：wubinxing
@Time：2021/4/29 13:49
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from ..env.xsim_manager import XSimManager
from ..env.communication_service import CommunicationService


class XSimEnv(object):
    """
        仿真环境类
        对于用户来说，如果想要与xsim环境连接，只需要实例化一个XSimEnv实例即可
        - 通过 step(action)->obs 将任务指令发送至xsim引擎，然后推进引擎，同时，引擎返回执行action后的observation
        - 通过 reset()重置xsim环境
        - 通过 close()关闭xsim环境
    @Examples:
        添加使用示例
        # 创建xsim环境
		xsim_env = XSimEnv()
		# 推进环境
		obs = xsim_env.step(action)
		# 重置环境
        env_reset_state = xsim_env.reset()
        # 关闭环境
        env_close_state = xsim_env.close()
    @Author：wubinxing
    """
    def __init__(self, time_ratio: int, address: str, image_name='bvrsim:v1.0', mode: str = 'host'):
        """
        初始化函数
        @param domain_id: 服务域名
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        # xsim引擎控制器
        self.xsim_manager = XSimManager(time_ratio, address, image_name, mode)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)

    def __del__(self):
        self.xsim_manager.close_env()

    def step(self, action: list) -> dict:
        """
        用户与xsim环境交互核心函数。通过step控制引擎的推进。
        @param action: 用户要执行的任务指令列表，任务指令可以通过EnvCmd任务指令组包辅助类进行辅助组包。
        @return: xsim在执行完毕action后当前的态势信息
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        try:
            obs = self.communication_service.step(action)
            return obs
        except Exception as e:
            print(e)
        # return self.communication_service.step(action)

    def reset(self):
        """
        重置训练环境
        @return: 环境重置状态：成功或者失败
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        return self.communication_service.reset()

    def end(self):
        """
        重置训练环境
        @return: 环境重置状态：成功或者失败
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        return self.communication_service.end()

    def close(self) -> bool:
        """
        关闭训练环境
        @return: 环境关闭状态：成功或者失败
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        self.xsim_manager.close_env()
        self.communication_service.close()
        return True
