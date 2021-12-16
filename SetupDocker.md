<style>
img{
    width: 30%;
    /* padding-left = (100% - width) / 2 */
    padding-left: 35%;
}
</style>
# start docker container
```bash
docker run -itd   --name  hmp \
--net host \
--gpus all \
--shm-size=16G \
fuqingxu/fqxdocker2021:latest1210
```

# make remote desktop running
```
rm -f /var/run/xrdp/xrdp-sesman.pid
service xrdp start
service xrdp status
```
Now, you should see both xrdp-sesman and xrdp are running via:
```
service xrdp status
```

Next, use the remote desktop tool of MS Windows (or anything supporting RDP) to get inside the HMP container.
![](2021-12-16-12-07-50.png)
```
IP: the IP of the host server running the docker. 
Port: 3389.
UserName: hmp
Password: hmp

(It's normal that xrdp is a bit slow, but there is no any better RDP solution for docker container yet,
therefore please use SSH when GUI is not needed)
```


# run HMP


```
# clone rep from gitee:
git clone https://gitee.com/hh505030475/hmp-2g.git

# or github (sync once a week, may not be the latest)
git clone https://github.com/binary-husky/hmp2g.git

# cd into it.
cd hmp-2g

# inspect experiment config
cat ./example.jsonc

# run experiment 
python main.py --cfg ./example.jsonc
```
<!-- ```
git clone git@gitee.com:hh505030475/hmp-2g.git
``` -->

