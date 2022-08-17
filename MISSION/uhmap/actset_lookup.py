import numpy as np

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 1, interface for RL # # # # # 
# # # # # # # # # # # # # # # # # # # # # # #

dictionary_items = [
    'ActionSet2::N/A;N/A',                  # 0
    'ActionSet2::Idle;DynamicGuard'     ,   # 1
    'ActionSet2::Idle;StaticAlert'      ,   # 2
    'ActionSet2::Idle;AggressivePersue' ,   # 3
    'ActionSet2::SpecificMoving;Dir+X'  ,   # 4
    'ActionSet2::SpecificMoving;Dir+Y'  ,   # 5
    'ActionSet2::SpecificMoving;Dir-X'  ,   # 6
    'ActionSet2::SpecificMoving;Dir-Y'  ,   # 7
    'ActionSet2::SpecificAttacking;T1-0',   # 8
    'ActionSet2::SpecificAttacking;T1-1',   # 9
    'ActionSet2::SpecificAttacking;T1-2',   # 10
    'ActionSet2::SpecificAttacking;T1-3',   # 11
    'ActionSet2::SpecificAttacking;T1-4',   # 12
    
    'ActionSet2::SpecificAttacking;T0-0',   # 13
    'ActionSet2::SpecificAttacking;T0-1',   # 14
    'ActionSet2::SpecificAttacking;T0-2',   # 15
    'ActionSet2::SpecificAttacking;T0-3',   # 16
    'ActionSet2::SpecificAttacking;T0-4',   # 17

    'ActionSet2::PatrolMoving;Dir+X'    ,
    'ActionSet2::PatrolMoving;Dir+Y'    ,
    'ActionSet2::PatrolMoving;Dir-X'    ,
    'ActionSet2::PatrolMoving;Dir-Y'    ,

    'ActionSet2::Idle;AsFarAsPossible',  
    'ActionSet2::Idle;StayWhenTargetInRange',  
    'ActionSet2::Idle;StayWhenTargetInHalfRange' ,  
]

dictionary_n_actions = len(dictionary_items)

digit2act_dictionary = {
    i: dictionary_items[i] for i, item in enumerate(dictionary_items)
}

act2digit_dictionary = {
    dictionary_items[i]:i for i, item in enumerate(dictionary_items)
}

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 2, translate actions # # # # # 
# # # # # # # # # # # # # # # # # # # # # # #

agent_json2local_attrs = [
            # json key       ----->    agent key
            ('agentAlive',              'alive'),
            ('agentTeam',               'team'),
            ('indexInTeam',             'index'),
            ('uId',                     'uid_remote'),
            ('maxMoveSpeed',            'max_speed'),
            ('agentLocationArr',        'location'),
            ('agentRotationArr',        'rotation'),
            ('agentScaleArr',           'scale3'),
            ('agentVelocityArr',        'velocity'),
            ('agentHp',                 'hp'),
            ('weaponCD',                'weapon_cd'),
            ('type',                    'type'),
]


    # 'ActionSet2::Idle;AsFarAsPossible',  
    # 'ActionSet2::Idle;StayWhenTargetInRange',  
    # 'ActionSet2::Idle;StayWhenTargetInHalfRange' ,  

def encode_action_as_digits(main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None):
    main_cmd_encoder = {
        "Idle"                  : 0, 
        "SpecificMoving"        : 1, 
        "PatrolMoving"          : 2, 
        "SpecificAttacking"     : 3, 
        "N/A"                   : 4, 
    }
    sub_cmd_encoder = {
        "DynamicGuard"              : 0 , 
        "StaticAlert"               : 1 , 
        "AggressivePersue"          : 2 , 
        "SpecificAttacking"         : 3 , 
        "AsFarAsPossible"           : 4 , 
        "StayWhenTargetInRange"     : 5 , 
        "StayWhenTargetInHalfRange" : 6 , 
        "N/A"                       : 7 , 
        'Dir+X'                     : 8 , 
        'Dir+X+Y'                   : 9 , 
        'Dir+Y'                     : 10,
        'Dir-X+Y'                   : 11,
        'Dir-X'                     : 12,
        'Dir-X-Y'                   : 13,
        'Dir-Y'                     : 14,
        'Dir+X-Y'                   : 15,
    }
    return np.array([
        main_cmd_encoder[main_cmd],
        sub_cmd_encoder[sub_cmd],
        x if x is not None                  else np.inf,
        y if y is not None                  else np.inf,
        z if z is not None                  else np.inf,
        UID if UID is not None              else np.inf,
        T if T is not None                  else np.inf,
        T_index if T_index is not None      else np.inf
    ])


def decode_action_as_string(digits):
    main_cmd_decoder = {
        0 :"Idle"                , 
        1 :"SpecificMoving"      , 
        2 :"PatrolMoving"        , 
        3 :"SpecificAttacking"   , 
        4 :"N/A"                 , 
    }
    sub_cmd_decoder =  {
        0  : "DynamicGuard"              , 
        1  : "StaticAlert"               , 
        2  : "AggressivePersue"          , 
        3  : "SpecificAttacking"         , 
        4  : "AsFarAsPossible"           , 
        5  : "StayWhenTargetInRange"     , 
        6  : "StayWhenTargetInHalfRange" , 
        7  : "N/A"                       , 
        8  : 'Dir+X'                     , 
        9  : 'Dir+X+Y'                   , 
        10 : 'Dir+Y'                     ,
        11 : 'Dir-X+Y'                   ,
        12 : 'Dir-X'                     ,
        13 : 'Dir-X-Y'                   ,
        14 : 'Dir-Y'                     ,
        15 : 'Dir+X-Y'                   ,
    }
    main_cmd = main_cmd_decoder[digits[0]]
    sub_cmd = sub_cmd_decoder[digits[1]]
    x       = digits[2] if np.isfinite(digits[2]) else None
    y       = digits[3] if np.isfinite(digits[3]) else None
    z       = digits[4] if np.isfinite(digits[4]) else None
    UID     = digits[5] if np.isfinite(digits[5]) else None
    T       = digits[6] if np.isfinite(digits[6]) else None
    T_index = digits[7] if np.isfinite(digits[7]) else None

    if main_cmd == "Idle":
        res = 'ActionSet2::Idle;%s'%sub_cmd
        assert res in dictionary_items, '指令错误无法解析'
    elif main_cmd == "SpecificMoving":
        if sub_cmd == 'N/A':
            res = 'ActionSet2::SpecificMoving;X=%f Y=%f Z=%f'%(x,y,z)
        else:
            res = 'ActionSet2::SpecificMoving;%s'%sub_cmd
    elif main_cmd == "PatrolMoving":
        if sub_cmd == 'N/A':
            res = 'ActionSet2::PatrolMoving;X=%f Y=%f Z=%f'%(x,y,z)
        else:
            res = 'ActionSet2::PatrolMoving;%s'%sub_cmd
    elif main_cmd == "SpecificAttacking":
        # 'ActionSet2::SpecificAttacking;T1-3',
        # 'ActionSet2::SpecificAttacking;UID-4',
        assert sub_cmd == 'N/A', '指令错误无法解析'
        if UID is not None:
            res = 'ActionSet2::SpecificAttacking;UID-%d'%UID
        else:
            res = 'ActionSet2::SpecificAttacking;T%d-%d'%(T,T_index)
    elif main_cmd == "N/A":
        res = 'ActionSet2::N/A;N/A'
    else:
        print('指令错误无法解析')
        assert False
    return res


# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 3, agent init defaults # # # #
# # # # # # # # # # # # # # # # # # # # # # #

AgentPropertyDefaults = {
    'ClassName': 'RLA_CAR',     # FString ClassName = "";
    'DebugAgent': False,     
    'AgentTeam': 0,             # int AgentTeam = 0;
    'IndexInTeam': 0,           # int IndexInTeam = 0;
    'UID': 0,                   # int UID = 0;
    'MaxMoveSpeed': 600,        # move speed, test ok
    'InitLocation': { 'x': 0,  'y': 0, 'z': 0, },
    'InitRotation': { 'x': 0,  'y': 0, 'z': 0, },
    'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },     # agent size, test ok
    'InitVelocity': { 'x': 0,  'y': 0, 'z': 0, },
    'AgentHp':100,
    "WeaponCD": 1,              # weapon fire rate
    "IsTeamReward": True,
    "Type": "",
    "DodgeProb": 0.8,           # probability of escaping dmg 闪避概率, test ok
    "ExplodeDmg": 25,           # ms explode dmg. test ok
    "FireRange": 1000.0,        # <= 1500
    "GuardRange": 1400.0,       # <= 1500
    "PerceptionRange": 1500.0,       # <= 1500
    'Color':'(R=0,G=1,B=0,A=1)',    # color
    "FireRange": 1000,
    'RSVD1':'',
    'RSVD2':'',
}



# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 3, framerate selection # # # #
# # # # # # # # # # # # # # # # # # # # # # #
# Check whether a number can be represented precisely by a float
def binary_friendly(x):
    y_f16 = np.array(x, dtype=np.float16)
    y_f64 = np.array(x, dtype=np.float64)
    t = y_f64 - y_f16
    assert t.dtype == np.float64
    return (t==0)


# '''
# <Agent>T0-5<UID>5<Event>Destroyed 
# <Agent>T1-6<UID>16<Event>Destroyed 
# <Agent>T0-4<UID>4<Event>Destroyed 
# <Agent>T0-2<UID>2<Event>Destroyed 
# <Agent>T1-7<UID>17<Event>Destroyed 
# <Agent>T1-1<UID>11<Event>Destroyed 
# <Agent>T0-8<UID>8<Event>Destroyed 
# <Agent>T0-7<UID>7<Event>Destroyed 
# <Agent>T1-3<UID>13<Event>Destroyed
# <Agent>T0-9<UID>9<Event>Destroyed
# <Agent>T0-6<UID>6<Event>Destroyed
# <Agent>T0-0<UID>0<Event>Destroyed
# <Agent>T1-2<UID>12<Event>Destroyed
# <Agent>T0-1<UID>1<Event>Destroyed
# <Agent>T0-3<UID>3<Event>Destroyed
# <Event>EndEpisode<EndReason><AllRLAgentDown>Lose<WinTeam>1
# '''





################## ########################## ########################
################## ########################## ########################
################## ########################## ########################
################## single digit encode, not used ########################
# h_map_center = (-7290.0, 6010.0)
# h_grid_size = 400
# v_ground = 340
# v_grid_size = 1000

# x_arr = np.array([h_map_center[0]+v_grid_size*i for i in range(-20, 20)])   # 0~39, 40, 1
# y_arr = np.array([h_map_center[1]+v_grid_size*i for i in range(-20, 20)])   # 0~39, 40, 40
# z_arr = np.array([v_ground+v_grid_size*i for i in range(4)])                # 0~3,  4,  1600
# # offset                                                                    # 0~1,  2,  6400


# # output $y \in [1000, 12800]$
# def _2digit(main_cmd, x, y, z):
#     z_logit = np.argmin(np.abs(z - z_arr))
#     x_logit = np.argmin(np.abs(x - x_arr))
#     y_logit = np.argmin(np.abs(y - y_arr))
#     if main_cmd=='SpecificMoving': cmd_logit = 0
#     elif main_cmd=='PatrolMoving': cmd_logit = 1
#     ls_mod = [1,40,1600,6400]
#     offset = 1000
#     x = np.array([x_logit, y_logit, z_logit, cmd_logit])
#     print(x)
#     y = np.dot(x, ls_mod)+offset
#     return y

# def _2coordinate(x):
#     offset = 1000
#     ls_mod = [1,40,1600,6400]
#     x = x - offset
#     res = []
#     for mod in reversed(ls_mod):
#         tmp = x // mod
#         x = x - tmp*mod
#         res.append(tmp)
#     res = list(reversed(res))
#     x_logit, y_logit, z_logit, cmd_logit = res
#     if    cmd_logit == 0 : main_cmd ='SpecificMoving'
#     elif  cmd_logit == 1 : main_cmd ='PatrolMoving'  
#     x = x_arr[x_logit]
#     y = y_arr[y_logit]
#     z = z_arr[z_logit]
#     print(main_cmd, x, y, z)
#     return main_cmd, x, y, z
################## ########################## ########################
################## ########################## ########################
################## ########################## ########################

    