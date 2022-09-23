# setup on ubuntu without docker

Warning! Always use docker if possible! 

如果可以选择，请一定先尝试docker方案

This file is written for those who are very confident to solve all kinds of problems and errors on their own!

本文件仅写给对解决各种软件依赖十分自信的老手作为参考

## python version

``` sh
python 3.8
```
## pip requirements 
``` sh
pip install numpy scipy
pip install lz4 gym flask numba cython waitress colorama func_timeout setproctitle 
pip install commentjson matplotlib psutil paramiko ipykernel onedrivedownloader flock
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## download and extract starcraft and unreal engine
``` sh
cd /home/hmp
git clone https://github.com/binary-husky/uhmap-visual-tool.git
cd /home/hmp/uhmap-visual-tool/
python linux_deploy.py
python linux_deploy_starcraft_all_versions.py
mv /home/hmp/uhmap-visual-tool/UnrealEngine/home/hmp/*  /home/hmp
``` 

## download and extract HMAP main framework
``` sh
cd /home/hmp
git clone https://github.com/binary-husky/hmp2g.git -b multiteam
cd /home/hmp/hmp2g
```
