# 该文件 用于定义 本文中使用的算法

import math

# 数学计算工具
class HRMathUtil:
    # 弧度转度
    @staticmethod
    def Rad2Deg(AngRad):
        return AngRad * 57.2957795131

    # 度转弧度
    @staticmethod
    def Deg2Rad(AngDeg):
        return AngDeg * 0.01745329252
#定义 一个 TSVector3D
class BaseTSVector3:
    # 初始化
    def __init__(self,x:float,y:float,z:float):
        return {"X": x, "Y":y, "Z": z}
    # 矢量a + 矢量b
    @staticmethod
    def plus(a, b):
        return {"X": a["X"] + b["X"], "Y": a["Y"] + b["Y"], "Z": a["Z"] + b["Z"]}

    # 矢量a - 矢量b
    @staticmethod
    def minus(a, b):
        return {"X": a["X"] - b["X"], "Y": a["Y"] - b["Y"], "Z": a["Z"] - b["Z"]}

    # 矢量a * 标量scal
    @staticmethod
    def multscalar(a, scal):
        return {"X": a["X"] * scal, "Y": a["Y"] * scal, "Z": a["Z"] * scal}

    # 矢量a / 标量scal
    @staticmethod
    def divdbyscalar(a, scal):
        if scal == 0:
            return {"X": 1.633123935319537e+16, "Y": 1.633123935319537e+16, "Z": 1.633123935319537e+16}
        else:
            return {"X": a["X"] / scal, "Y": a["Y"] / scal, "Z": a["Z"] / scal}

    # 矢量a 点乘 矢量b
    @staticmethod
    def dot(a, b):
        return a["X"] * b["X"] + a["Y"] * b["Y"] + a["Z"] * b["Z"]

    # 矢量a 叉乘 矢量b
    @staticmethod
    def cross(a, b):
        val = {"X": a["Y"] * b["Z"] - a["Z"] * b["Y"], \
               "Y": a["Z"] * b["X"] - a["X"] * b["Z"], \
               "Z": a["X"] * b["Y"] - a["Y"] * b["X"]}
        return val

    # 判断矢量a是否为0矢量
    @staticmethod
    def iszero(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return True
        else:
            return False

    # 矢量a归一化
    @staticmethod
    def normalize(a):
        vallen = math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])
        val = {"X": 0, "Y": 0, "Z": 0}
        if vallen > 0:
            val = {"X": a["X"] / vallen, "Y": a["Y"] / vallen, "Z": a["Z"] / vallen}
        return val

    # 计算矢量a的长度
    @staticmethod
    def length(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return 0
        return math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])

    # 计算矢量a的长度平方
    @staticmethod
    def lengthsqr(a):
        return a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"]
# 三维矢量计算工具
class TSVector3(BaseTSVector3):
    #初始化
    def __init__(self,x:float,y:float,z:float):
        return {"X": x, "Y":y, "Z": z}
    # 计算位置矢量a与位置矢量b间的距离
    @staticmethod
    def distance(a, b):
        return BaseTSVector3.length(BaseTSVector3.minus(a, b))

    # 计算位置矢量a与位置矢量b间的距离平方
    @staticmethod
    def distancesqr(a, b):
        return BaseTSVector3.lengthsqr(BaseTSVector3.minus(a, b))

    # 计算矢量a与矢量b之间的夹角，单位弧度
    @staticmethod
    def angle(a, b):
        if BaseTSVector3.iszero(a) or BaseTSVector3.iszero(b):
            return 0
        else:
            ma = BaseTSVector3.length(a)
            mb = BaseTSVector3.length(b)
            mab = BaseTSVector3.dot(a, b)
        return math.acos(mab / ma / mb)

    # 给定方位角heading和俯仰角pitch，单位弧度，计算单位方向矢量
    @staticmethod
    def calorientation(heading, pitch):
        return {"X": math.sin(heading) * math.cos(pitch), "Y": math.cos(heading) * math.cos(pitch),
                "Z": math.sin(pitch)}

    # 计算矢量direction的方位角，单位弧度
    @staticmethod
    def calheading(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        else:
            heading = math.atan2(direction["X"], direction["Y"])
            if heading < 0:
                heading += math.pi * 2
            return heading

    # 计算矢量direction的方位角，单位度
    @staticmethod
    def calheading_deg(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        else:
            heading = math.atan2(direction["X"], direction["Y"])
            if heading < 0:
                heading += math.pi * 2
            return HRMathUtil.Rad2Deg(heading)

    # 计算矢量direction的俯仰角，单位弧度
    @staticmethod
    def calpitch(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        elif direction["X"] == 0 and direction["Y"] == 0:
            return math.pi * 0.5
        elif direction["Z"] == 0:
            return 0
        else:
            mxy = math.sqrt(direction["X"] * direction["X"] + direction["Y"] * direction["Y"])
            return math.atan2(direction["Z"], mxy)

    # 计算矢量direction的俯仰角，单位度
    @staticmethod
    def calpitch_deg(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        elif direction["X"] == 0 and direction["Y"] == 0:
            return math.pi * 0.5
        elif direction["Z"] == 0:
            return 0
        else:
            mxy = math.sqrt(direction["X"] * direction["X"] + direction["Y"] * direction["Y"])
            return HRMathUtil.Rad2Deg(math.atan2(direction["Z"], mxy))

    # 计算位置矢量pos1与位置矢量pos2之间的地面距离
    @staticmethod
    def groundrange(pos1, pos2):
        return math.sqrt((pos1["X"] - pos1["X"]) * (pos1["X"] - pos1["X"]) + \
                         (pos1["Y"] - pos1["Y"]) * (pos1["Y"] - pos1["Y"]))

