# how to build: docker build --network=host -progress=plain -t hmp-from-scrach . 

# 此文件用于构建 HMAP + Starcraft(难易两个版本) + 虚幻引擎组件
# 可能需要翻墙，请自行搭梯子，然后解开下面相关的注释，并进行适当修改（socks5地址和端口）

FROM nvidia/cuda:11.5.1-runtime-ubuntu20.04
RUN apt-get update

RUN apt-get install -y curl proxychains
RUN $useProxyNetwork curl cip.cc


ENV TZ=Asia/Shanghai
ENV LC_ALL zh_CN.UTF-8
RUN apt-get install -y language-pack-zh-hans \
    libmysqlclient-dev \
    dialog \
    nano \
    vim \
    joe \
    wget \
    curl \
    jq \
    gawk \
    psmisc \
    python \
    python3 \
    python-yaml \
    python-jinja2 \
    python3-urllib3 \
    python-tz \
    python-nose \
    python3-prettytable \
    python-netifaces \
    python-dev \
    python3-pip \
    python3-mysqldb \
    openjdk-8-jre \
    openjdk-8-jdk \
    openssh-server \
    openssh-client \
    git \
    sudo \
    inotify-tools \
    rsync \
    net-tools \
    cron \
    swig \
    cmake \
    redis-tools \
    redis-server\
    iproute2 \
    pkg-config build-essential libssl-dev libffi-dev --fix-missing

RUN locale-gen zh_CN.UTF-8 && localedef -c -f UTF-8 -i zh_CN zh_CN.utf8 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# create user and add to sudoers
RUN useradd -m hmp && echo "hmp:hmp" | chpasswd && adduser hmp sudo
USER hmp
CMD /bin/bash
# RUN echo hmp|sudo -S apt-get install -y nano
WORKDIR /home/hmp

# use python3 as the system default python
USER root
RUN rm /usr/bin/python
RUN ln /usr/bin/python3 /usr/bin/python
ARG useProxyNetwork=''
RUN $useProxyNetwork curl cip.cc
# # comment out below if you do not need proxy network | 翻墙 - 从此行向下删除
# RUN sed -i '$ d' /etc/proxychains.conf
# RUN sed -i '$ d' /etc/proxychains.conf
# RUN echo "socks5 127.0.0.1 10880" >> /etc/proxychains.conf
# ARG useProxyNetwork=proxychains
# RUN $useProxyNetwork curl cip.cc
# # comment out above if you do not need proxy network | 翻墙 - 从此行向上删除

# pip install everything we need
USER hmp
SHELL ["/bin/bash", "-c"]
RUN $useProxyNetwork pip install numpy scipy scikit-learn
RUN $useProxyNetwork pip install lz4 gym flask cython waitress colorama func_timeout setproctitle filelock
RUN $useProxyNetwork pip install commentjson matplotlib psutil paramiko ipykernel onedrivedownloader flock
RUN $useProxyNetwork pip install torch --extra-index-url https://download.pytorch.org/whl/cu115
RUN $useProxyNetwork pip install pygame cmake redis numba
RUN $useProxyNetwork pip install git+https://github.com/oxwhirl/smac.git

# # download and extract UHMAP component
# WORKDIR /home/hmp
# RUN $useProxyNetwork git clone https://github.com/binary-husky/uhmap-visual-tool.git
# WORKDIR /home/hmp/uhmap-visual-tool/
# RUN $useProxyNetwork python linux_deploy.py
# RUN $useProxyNetwork python linux_deploy_starcraft_all_versions.py
# # add execute mod to starcraft II (two versions)
# RUN chmod +x /home/hmp/StarCraftIII/Version2410/StarCraftII/Versions/Base75689/ 
# RUN chmod +x /home/hmp/StarCraftIII/Versions/Base69232/SC2_x64

# RUN mv /home/hmp/uhmap-visual-tool/UnrealEngine/home/hmp/*  /home/hmp

# download UHMAP main framwork
WORKDIR /home/hmp
RUN $useProxyNetwork git clone https://github.com/binary-husky/hmp2g.git
WORKDIR /home/hmp/hmp2g
# RUN python main.py -c example.jsonc

# # Installing Times New Roman font
# USER root
# # RUN apt-get --reinstall install ttf-mscorefonts-installer
# # RUN apt-get install msttcorefonts -qq
# # RUN rm /home/hmp/.cache/matplotlib -rf

USER root

# RUN sed -i 's/22/2233/g' /etc/ssh/sshd_config
RUN echo "Port 2233" >> /etc/ssh/sshd_config
RUN echo "service ssh start" >> /entrypoint.sh
RUN echo "redis-server --daemonize yes" >> /entrypoint.sh


# exit
USER hmp
WORKDIR /home/hmp
RUN touch /home/hmp/.sudo_as_admin_successful
COPY ./bashrc_suffix /home/hmp/bashrc_suffix
RUN cat /home/hmp/bashrc_suffix >> /home/hmp/.bashrc

# docker build --network=host --progress=plain -t py38 .
# docker run -itd  --name  hmp-fqx --net host --gpus all --shm-size=16G py38:latest && docker exec -it -u 0 hmp-fqx  service ssh start
# docker exec -it hmp-fqx bash
# docker stop hmp-fqx && docker rm hmp-fqx