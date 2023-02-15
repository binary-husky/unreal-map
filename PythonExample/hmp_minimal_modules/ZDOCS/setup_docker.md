
# 1. Install nvidia docker runtime
Cuda is needed inside our docker container, which need toolkits from Nvidia for GPU support.
Please install nvidia docker runtime on the host ubuntu system.

For details, refer to nvidia official document: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian

According the link above, we write a manual about installing nvidia docker runtime:
Please read [SetupUbuntu](./setup_ubuntu.md).



# 2. Start docker container
From the host:
```bash
$ docker run -itd   --name  hmp-$USER \
--net host \
--gpus all \
--shm-size=16G \
fuqingxu/hmp:latest
```
Warning! Need at least 50GB disk space because cuda, Starcraft environment and all needed python package is packed inside.

Warning! we use ```--net host``` to bridge the docker container for a lot of convenience.

Unpredictable errors may occur if the port inside container conflict with the host network, e.g. port 3389(rdp), 6379(redis), 2233(ssh), make sure the host system is not using them!

Unpredictable errors may occur if you decide to use ```-p``` parameter to mount other ports.

Finally check docker status with ```docker ps```, should be seeing a container named ```hmp``` at running state.



# (3. Optional) Get inside HMP container via SSH
```
$ docker exec -it hmp-$USER service ssh start
```

Now find a computer to ssh into it: ```ssh hmp@your_host_ip -p 2233```
```
# IP Addr: share with the host
# SSH Port 2233
# UserName: hmp
# Password: hmp
```

Note: The environment is not configured in the ```root``` account! 
If you enter directly after ```docker run``` (not using ssh), 
you have to switch the account manually from ```root``` to ```hmp``` (using linux command ```su hmp```), 


# (3. Optional) Connect to HMP container with remote desktop (RDP)
(choice 1) Use SSH to get ```inside``` the HMP container.

(choice 2) From the host, use ``` docker exec -it hmp-$USER bash ``` command to get inside the HMP container.

Then:
```sh
# before continue, make sure the host port 3389 is free to use for RDP

(hmp-docker)$ sudo /etc/init.d/xrdp stop; sleep 5;
(hmp-docker)$ sudo rm -rf /var/run/xrdp/xrdp-sesman.pid; sleep 5;
(hmp-docker)$ sudo xrdp; sleep 5;
(hmp-docker)$ sudo /etc/init.d/xrdp start; sleep 5;
```
Now, you should see xrdp-sesman running via:
```sh
(hmp-docker)$ /etc/init.d/xrdp status

# Successful if you see >>
#   * xrdp-sesman is running
#   * xrdp is running

# note: if multiple instances of hmp-docker is running,
# you should modify following settings to avoid port collision into some value that is not default:
#  /etc/xrdp/sesman.ini: The 'X11DisplayOffset' and 'ListenPort' option
#  /etc/xrdp/xrdp.ini: The 'port' option
```



Next, use the remote desktop tool of MS Windows (or anything supporting RDP) to get inside the HMP container.
```
# IP Addr: share with the host
# RDP Port: 3389.
# UserName: hmp
# Password: hmp

(It's normal that xrdp is a bit slow, but there is no better RDP solution for docker container yet, please use SSH when GUI is not needed)
```

# 4. Run HMP
After getting ```inside``` the HMP container:

```
# if current user is 'root', change to a user with name 'hmp' (password also 'hmp'):
(hmp-container)$ su hmp

# goto its home directory
(hmp-container)$ cd

# clone rep from github:
(hmp-container)$ git clone https://github.com/binary-husky/hmp2g.git

# or gitee (sync once a week, may not be the latest, please use gitee rep if possible)
(hmp-container)$ git clone https://gitee.com/hh505030475/hmp-2g.git

# cd into it.
(hmp-container)$ cd hmp2g

# run an trained model to find out if everthing works well ^_^
(hmp-container)$ git pull && python main.py -c ZHECKPOINT/50RL-55opp/test-50RL-55opp.jsonc

```
<img src="../ZHECKPOINT/test-50+50/butterfly.webp" width="200" >

# Docker in Docker (If need to run air combat env)

If you want to play ```docker in docker```, please mount ```/var/run/docker.sock```:
```bash
docker run -itd   --name  hmp-$USER \
--volume /var/run/docker.sock:/var/run/docker.sock \
--net host \
--gpus all \
--shm-size=16G \
fuqingxu/hmp:latest
```
<img src="../ZHECKPOINT/test-50+50/butterfly.webp" width="200" >

# Appendix：requirement.txt (install on Windows)
If possible, please ```use docker``` to Avoid following
pip package management.
This requirement list is provided only as 
a reminder of dependencies being used,
```do NOT use it for configuration unless no other choice is available!```


Please read [pip_requirement](pip_requirement.md)
