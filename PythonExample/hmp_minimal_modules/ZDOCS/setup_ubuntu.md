# 1. install docker 
```
sudo apt update
sudo apt install docker docker.io curl
```

# 2. intsall nvidia-runtime 

``` sh
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian

# step 1
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

# step 2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# step 3:Install the nvidia-docker2 package (and dependencies) after updating the package listing:

sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to complete the installation after setting the default runtime:

sudo systemctl restart docker
```


# 3. Download docker image and open it 

In this part, please read [SetupDocker](./setup_docker.md)
