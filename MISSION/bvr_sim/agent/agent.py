"""
@FileName：agent.py
@Description：agent基类文件
@Author：liyelei
@Time：2021/4/29 11:24
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from typing import List


class Agent(object):
    """
        用于框架中进行训练的算法的基类，选手通过继承该类，去实现自己的策略逻辑
    @Examples:
        添加使用示例
		>>> 填写使用说明
		··· 填写简单代码示例
    @Author：liyelei
    """
    def __init__(self, name, side, **kwargs):
        """必要的初始化"""
        self.name = name
        self.side = side

    def reset(self, **kwargs):
        pass

    def step(self, **kwargs) -> List[dict]:
        """输入态势信息，返回指令列表"""
        raise NotImplementedError

