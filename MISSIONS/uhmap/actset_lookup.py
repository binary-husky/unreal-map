dictionary_items = [
    'ActionSet2::Idle;DynamicGuard'     ,
    'ActionSet2::Idle;StaticAlert'      ,
    'ActionSet2::Idle;AggressivePersue' ,
    'ActionSet2::SpecificMoving;Dir+X'  ,
    'ActionSet2::SpecificMoving;Dir+X+Y',
    'ActionSet2::SpecificMoving;Dir+Y'  ,
    'ActionSet2::SpecificMoving;Dir-X+Y',
    'ActionSet2::SpecificMoving;Dir-X'  ,
    'ActionSet2::SpecificMoving;Dir-X-Y',
    'ActionSet2::SpecificMoving;Dir-Y'  ,
    'ActionSet2::SpecificMoving;Dir+X-Y',
    'ActionSet2::SpecificMoving;X=-6800.0 Y=6830.0 Z=397.215454',
    'ActionSet2::PatrolMoving;Dir+X'    ,
    'ActionSet2::PatrolMoving;Dir+X+Y'  ,
    'ActionSet2::PatrolMoving;Dir+Y'    ,
    'ActionSet2::PatrolMoving;Dir-X+Y'  ,
    'ActionSet2::PatrolMoving;Dir-X'    ,
    'ActionSet2::PatrolMoving;Dir-X-Y'  ,
    'ActionSet2::PatrolMoving;Dir-Y'    ,
    'ActionSet2::PatrolMoving;Dir+X-Y'  ,
    'ActionSet2::PatrolMoving;X=-6800.0 Y=6830.0 Z=397.215454',
    'ActionSet2::SpecificAttacking;T1-1',
    'ActionSet2::SpecificAttacking;T1-2',
    'ActionSet2::SpecificAttacking;T1-3',
    'ActionSet2::SpecificAttacking;T1-4',
    'ActionSet2::N/A;N/A',
    'ActionSet2::SpecificAttacking;UID-0',
    'ActionSet2::SpecificAttacking;UID-1',
    'ActionSet2::SpecificAttacking;UID-2',
    'ActionSet2::SpecificAttacking;UID-3',
    'ActionSet2::SpecificAttacking;UID-4',
    'ActionSet2::SpecificAttacking;UID-5',
    'ActionSet2::SpecificAttacking;UID-6',
    'ActionSet2::SpecificAttacking;UID-7',
    'ActionSet2::SpecificAttacking;UID-8',
]

dictionary_n_actions = len(dictionary_items)

digit2act_dictionary = {
    i: dictionary_items[i] for i, item in enumerate(dictionary_items)
}

act2digit_dictionary = {
    dictionary_items[i]:i for i, item in enumerate(dictionary_items)
}

no_act_placeholder = 'ActionSet2::N/A;N/A'

agent_json2local_attrs = [
            # json key       ----->    agent key
            ('agentAlive',              'alive'),
            ('agentTeam',               'team'),
            ('indexInTeam',             'index'),
            ('uId',                     'uid_remote'),
            ('acceptRLControl',         'rl_agent'),
            ('maxMoveSpeed',            'max_speed'),
            ('agentLocation',           'Location'),
            ('agentRotation',           'Rotation'),
            ('agentScale',              'scale'),
            ('agentVelocity',           'vel'),
            ('agentHp',                 'hp'),
            ('weaponCD',                'weaponCD'),
]