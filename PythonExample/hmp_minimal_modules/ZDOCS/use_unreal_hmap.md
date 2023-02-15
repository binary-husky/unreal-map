

# Get Unreal-HMAP Binary Client (Win & Linux)

- Method 1: 
``` python 
from MISSION.uhmap.auto_download import download_client_binary_on_platform
download_client_binary_on_platform(
    desired_path="./UnrealHmapBinary/Version3.5/LinuxNoEditor/UHMP.sh", 
    # desired_path="./UnrealHmapBinary/Version3.5/LinuxNoEditor/UHMP.exe", 
    desired_version="3.5", 
    is_render_client=True,
    platform="Linux",
    # platform="Windows",

    
)
```

- Method 2 (manual): download uhmap file manifest (a json file)
```
https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/EVmCQMSUWV5MgREWaxiz_GoBalBRV3DWBU3ToSJ5OTQaLQ?e=I8yjl9
```
Open this json file, choose the version and platform you want, download and unzip it.

- Method 3 (Compile from source): 
```
https://github.com/binary-husky/unreal-hmp
```
