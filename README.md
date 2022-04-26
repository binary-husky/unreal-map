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
## Web Demo

```
http://cloud.fuqingxu.top:11601/
```
<img src="ZHECKPOINT/test-50+50/test50.gif" width="300" >


## Anti-Invasion Interception
```
git pull && python main.py -c ZHECKPOINT/test-aii515/test-aii515.jsonc --skip 
```

## Hazardous Cargo Transport
```
git pull && python main.py -c ZHECKPOINT/test-cargo50/test-cargo50.jsonc --skip
```

## Decentralized Collective Assult
```
git pull && python main.py -c ZHECKPOINT/test-50+50/test-50+50.jsonc --skip
git pull && python main.py -c ZHECKPOINT/test-100+100/test-100+100.jsonc --skip
```

# Dependency
We use docker to solve dependency: [SetupDocker](./setup_docker.md)


# Introducing the structure of HMP

## HMP's config system (How to experiment)
Unfinished doc

## Task runner
Unfinished doc

## ALGORITHM
Unfinished doc
### The time sequence of hmp-2g
<img src="UTILS/hmp2g_timeline.svg" width="700" >


## MISSIONS
Unfinished doc

## Execution Pool
Unfinished doc

## VHMAP, a component of HMP
VHMAP is a visulization component of HMP. [VHMAP](./VISUALIZE/README.md)

It is unfortunate that 
all existing RL environments fails to provide a visual
interface satisfying following useful features:

- Allowing visualizing while training, without slowing down the training server. 
- Using as little resourse as possible.
- Friendly to SSH users, faster than RDP and X server which is notoriously slow.
- No dependency, even an Android with a browser can access.
- Smooth, using client's CPU and GPU to render instead of server's.
- Simple. No verbose lines about lights, buffering, reflesh and bla.bla.bla about which we researchers never care.

VHMAP is just the answer,Features:
- Python interface simplified to the max
- Rendering on the client side, automatic frame insertion, and silky smooth frame rates
- Few server-side dependencies
- Very low server-side resource consumption
- Based on ThreeJs, drag and drop support, mobile touch screen support
- Support switching between perspective and projection views
- Playback support
- Use zlib to compress data streams, low network bandwidth requirement

<div align="center">
<img src="VISUALIZE/md_imgs/动画9.gif" width="700" >
</div>

Interface functions, operation introduction.
- Right mouse button to pan, left mouse button to rotate, scroll wheel to zoom
- Support touch screen, if your laptop or phone has a touch screen
- Rendering refresh rate is displayed in the upper left corner
- play fps: how many key frames per second (less than the rendering refresh rate, then insert frames; greater than the rendering refresh rate, then the excess is invalid)
- pause: pause
- next frame: pause and switch the next frame
- previous frame: pause and switch the previous frame
- loop to start: play all data, go back to the first frame
- ppt step: play a frame at a very slow speed, easy to record the screen, will be stuck for a few seconds after pressing
- use orthcam: switch the perspective view (object near large and far small)/projection view (similar to what you would use in engineering drawing)
- P.S. The first time you switch to the projection view, you need to use the mouse wheel to enlarge the screen



# Quick Start

## 0. dependency
We use docker to solve dependency: 
[SetupDocker](./SetupDocker.md)

Please read SetupDocker.md, then set up the container using:
```bash
$ docker run -itd   --name  hmp-$USER \
--net host \
--gpus all \
--shm-size=16G \
fuqingxu/hmp:latest
```

## 1. all default: testing
```
git pull && python main.py -c ZHECKPOINT/test-50+50/test-50+50.jsonc --skip
git pull && python main.py -c ZHECKPOINT/test-100+100/test-100+100.jsonc --skip
```
When the testing starts, open revealed url for monitoring. The front end is done by JavaScript and ThreeJS.
```
--------------------------------
JS visualizer online: http://172.18.116.150:aRandomPort
JS visualizer online (localhost): http://localhost:aRandomPort
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

# How to add a new environment (MISSION) in HMP
- make a new jsonc config file, using 'example.jsonc' as template
- mkdir in MISSIONS, e.g. ./MISSIONS/bvr_sim, copy src code of the environment inside it.
- open ```MISSIONS/env_router.py```, add the path of environment's init function in ```env_init_function_ref```, e.g.:
``` python
env_init_function_ref = {
    "bvr": ("MISSIONS.bvr_sim.init_env", "ScenarioConfig"),
}   
# bvr is the final name that HMP recognize, 
# MISSIONS.bvr_sim.init_env is a py file, 
# ScenarioConfig is a class
```
- open ```MISSIONS/env_router.py```, add the path of environment's configuration in ```import_path_ref```
``` python
import_path_ref = {
    "bvr": ("MISSIONS.bvr_sim.init_env", 'make_bvr_env'),
}   
# bvr is the final name that HMP recognize, 
# MISSIONS.bvr_sim.init_env is a py file, 
# make_bvr_env is a function
```
- write your own ScenarioConfig. (refer to ```MISSIONS.bvr_sim.init_env.ScenarioConfig```, as a template).
- write your own env init function. (refer to ```MISSIONS.bvr_sim.init_env.make_bvr_env```, as a template).





# Papers Supported by HMP

```
<1> Qingxu, F.; Tenghai, Q.; Jianqiang, Y.; Zhiqiang, Q.; and Shiguang, W. 2022. Concentration Network for Reinforcement Learning of Large-Scale Multi-Agent Systems. In Proceedings of the AAAI Conference on Artificial Intelligence
```
