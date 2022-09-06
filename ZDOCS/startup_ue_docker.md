# start up docker container
```sh
# 检查docker是否可用 （如果已经身处某个docker容器内，则docker不可用，请找到宿主系统，然后再运行以下命令）
sudo docker ps
```

```sh
# 启动docker容器
sudo docker run -itd   --name  $USER-swarm \
--net host \
--memory 500G \
--gpus all \
--shm-size=32G \
fuqingxu/hmp:unreal-trim


# 修改docker容器的ssh的端口到 4567，自行选择合适的空闲端口
sudo docker exec -it  $USER-swarm sed -i 's/2266/4567/g' /etc/ssh/sshd_config
# 运行docker容器的ssh
sudo docker exec -it  $USER-swarm service ssh start
# 运行docker容器的bash
sudo docker exec -it  $USER-swarm bash
```

Now find a computer to ssh into it: ```ssh hmp@your_host_ip -p 2233```
```
# IP Addr: share with the host
# SSH Port 4567
# UserName: hmp
# Password: hmp
```
