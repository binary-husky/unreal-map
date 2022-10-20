# how to build: docker build --network=host -progress=plain -t hmp-from-scrach . 

# 此文件用于构建 HMAP + Starcraft(难易两个版本) + 虚幻引擎组件
# 可能需要翻墙，请自行搭梯子，然后解开下面相关的注释，并进行适当修改（socks5地址和端口）

FROM nvidia/cuda:11.5.1-runtime-ubuntu20.04
RUN apt-get update
ARG useProxyNetwork=''

RUN apt-get install -y curl proxychains
RUN $useProxyNetwork curl cip.cc

# # comment out below if you do not need proxy network | 翻墙 - 从此行向下删除
# RUN sed -i '$ d' /etc/proxychains.conf
# RUN sed -i '$ d' /etc/proxychains.conf
# RUN echo "socks5 127.0.0.1 10880" >> /etc/proxychains.conf
# ARG useProxyNetwork=proxychains
# RUN $useProxyNetwork curl cip.cc
# # comment out above if you do not need proxy network | 翻墙 - 从此行向上删除

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
    python3 \
    redis-tools \
    python3-pip \
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

# pip install everything we need
USER hmp
SHELL ["/bin/bash", "-c"]
RUN $useProxyNetwork pip install numpy scipy
RUN $useProxyNetwork pip install lz4 gym flask numba cython waitress colorama func_timeout setproctitle 
RUN $useProxyNetwork pip install commentjson matplotlib psutil paramiko ipykernel onedrivedownloader flock
RUN $useProxyNetwork pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# download and extract UHMAP component
WORKDIR /home/hmp
RUN $useProxyNetwork git clone https://github.com/binary-husky/uhmap-visual-tool.git
WORKDIR /home/hmp/uhmap-visual-tool/
RUN $useProxyNetwork python linux_deploy.py
RUN $useProxyNetwork python linux_deploy_starcraft_all_versions.py
RUN mv /home/hmp/uhmap-visual-tool/UnrealEngine/home/hmp/*  /home/hmp

# download UHMAP main framwork
WORKDIR /home/hmp
RUN $useProxyNetwork git clone https://github.com/binary-husky/hmp2g.git -b multiteam
WORKDIR /home/hmp/hmp2g
# RUN python main.py -c example.jsonc

# Installing Times New Roman font
USER root
RUN apt-get --reinstall install ttf-mscorefonts-installer



# exit
USER hmp
WORKDIR /home/hmp