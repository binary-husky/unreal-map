<style>
img{
    width: 30%;
    /* padding-left = (100% - width) / 2 */
    padding-left: 35%;
}
</style>
# Install nvidia docker runtime
Cuda is needed inside our docker container, which need toolkits from Nvidia for GPU support.
Please install nvidia docker runtime on the host ubuntu system.

For details, refer to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian

# Start docker container
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

# Get inside HMP container via SSH
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




# (Optional) Connect to HMP container with remote desktop (RDP)
(choice 1) Use SSH to get ```inside``` the HMP container.

(choice 2) From the host, use ``` docker exec -it hmp-$USER bash ``` command to get inside the HMP container.

Then:
```
(hmp-docker)$ /etc/init.d/xrdp stop; sleep 5;
(hmp-docker)$ rm -rf /var/run/xrdp/xrdp-sesman.pid; sleep 5;
(hmp-docker)$ xrdp; sleep 5;
(hmp-docker)$ /etc/init.d/xrdp start; sleep 5;
```
Now, you should see xrdp-sesman running via:
```
(hmp-docker)$ /etc/init.d/xrdp status

# Successful if you see >>
#   * xrdp-sesman is running
#   * xrdp is running
```

Next, use the remote desktop tool of MS Windows (or anything supporting RDP) to get inside the HMP container.
```
# IP Addr: share with the host
# RDP Port: 3389.
# UserName: hmp
# Password: hmp

(It's normal that xrdp is a bit slow, but there is no better RDP solution for docker container yet, please use SSH when GUI is not needed)
```

# Run HMP
After getting ```inside``` the HMP container:

```
# if current user is 'root', change to a user with name 'hmp' (password also 'hmp'):
(hmp-container)$ su hmp

# goto its home directory
(hmp-container)$ cd

# clone rep from gitee:
(hmp-container)$ git clone https://gitee.com/hh505030475/hmp-2g.git

# or github (sync once a week, may not be the latest)
(hmp-container)$ git clone https://github.com/binary-husky/hmp2g.git

# cd into it.
(hmp-container)$ cd hmp-2g

# inspect experiment config
(hmp-container)$ cat ./example.jsonc

# run experiment 
(hmp-container)$ python main.py --cfg ./example.jsonc
```
<!-- ```
git clone git@gitee.com:hh505030475/hmp-2g.git
``` -->

