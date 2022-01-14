Author: Fu Qingxu, CASIA
# Introduction
Hybrid Multi-agent Playground (HMP) is an experimental framework designed for RL researchers.
Unlike any other framework which only isolates the TASKs from the framework, 
HMP also separates the ALGORITHMs from the framework to achieve excellent compatibility.

Any algorithm, from the most straightforward script AI to sophisticated RL learner,
is abstracted into a module inside ./ALGORITHM/*.

We also put effect to interface all kinds of multi-agent environments,
including gym, SMAC, air combat, et.al.

Other frameworks such as pymarl2 can interface with HMP as well.
The entire HMP can disguise as an RL environment in pymarl2.
We make it happen by building a particular ALGORITHM module, which
runs pymarl2 in a subprocess. This work is ongoing and has not been finished yet.

The root rep url is ```https://gitee.com/hh505030475/hmp-2g.git```,
we also have a github rep which is a mirror of this gitee rep. 
# Demo
```
http://home.fuqingxu.top:11601/     (ipv6 network only, can't afford a stable ipv4 address)
http://linux.ipv4.fuqingxu.top:11601/     (temporary ipv4)
```

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
## 2. all default: training

```
git pull && python main.py -c example.jsonc
```


## 3. change settings

launch with: 
```
python main.py --cfg xx.json
```

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

# Papers Supported by HMP

```
<1> Qingxu, F.; Tenghai, Q.; Jianqiang, Y.; Zhiqiang, Q.; and Shiguang, W. 2022. Concentration Network for Reinforcement Learning of Large-Scale Multi-Agent Systems. In Proceedings of the AAAI Conference on Artificial Intelligence
```
