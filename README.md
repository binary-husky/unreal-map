Author: Fu Qingxu, CASIA
# Introduction
Hybrid Multi-agent Playground (HMP) is a experimental framework designed for RL researchers.
Unlike any other framework, 
HMP completely isolate algorithm from the framework.

Any algorithm, from the simplist script AI to sophisticated RL learner,
is abstracted into a module inside ./ALGORITHM/*.

We also put effect to interface all kinds of multi-agent environmnets,
including gym, SMAC, air combat, et.al.

Other framework such as pymarl2 can interface with HMP as well.
We make it happen by building a special ALGORITHM module that 
run pymarl2 in a subprocess, and pretends itself to be a RL environment.
This work is on going and has not finished yet.

The root rep url is ```https://gitee.com/hh505030475/hmp-2g.git```,
we also have a github rep which is a mirror of this gitee rep. 

# Quick Start

## 0. dependency
We use docker to solve dependency: 
[SetupDocker](./SetupDocker.md)


## 1. all default: testing
```
git pull && python main.py -c ZHECKPOINT/test-50+50/test-50+50.jsonc --skip
git pull && python main.py -c ZHECKPOINT/test-100+100/test-100+100.jsonc --skip
```
When the testing starts, open revealed url for monitoring. The front end is done by JavaScript and ThreeJS.
```
--------------------------------
JS visualizer online: http://172.18.116.150:????
JS visualizer online (localhost): http://localhost:????
--------------------------------
```
## 2. all default: training (outdated, will soon update)

```
git pull && python main.py -c example.jsonc
```


## 3. change settings

launch with: 
```
python main.py --cfg xx.json
```

python main.py -c ZHECKPOINT/50vs50-eval/test.json
git pull && python main.py -c ZHECKPOINT/10

# Project Roadmap
If you are interested in something, you may continue to read:
```
    Handling parallel environment             -->   task_runner.py & shm_env.py

    Link between teams and diverse algorithms -->   multi_team.py

    Adding new env                            -->   MISSIONS.env_router.py

    Adding algorithm                          -->   ALGORITHM.example_foundation.py

    Configuring by writing py files           -->   config.py

    Configuring by json                       -->   xx.json

    colorful printing                         -->   colorful.py

    auto pip deployer                         -->   pip_find_missing.py

    efficient parallel execting               -->   shm_pool.pyx

    auto gpu selection                        -->   auto_gpu.py

    matlab logging/plotting bridge            -->   mcom.py & mcom_rec.py

    experiment batch executor                 -->   mprofile.py
```

