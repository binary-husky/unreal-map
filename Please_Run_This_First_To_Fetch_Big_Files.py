import os, commentjson, shutil, subprocess, tqdm, shutil, distutils
from onedrivedownloader import download

try:
    os.makedirs('./TEMP')
except:
    shutil.rmtree('./TEMP')
    os.makedirs('./TEMP')

def download_from_shared_server(key = 'cat'):
    # download uhmap file manifest | 下载manifest目录文件
    print('download uhmap file manifest | 下载manifest目录文件')
    manifest_url = "https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/EVmCQMSUWV5MgREWaxiz_GoBalBRV3DWBU3ToSJ5OTQaLQ?e=I8yjl9"

    try:
        file = download(manifest_url, filename="./TEMP/")
    except:
        print('failed to connect to onedrive | 连接onedrive失败, 您可能需要翻墙才能下载资源')

    with open("./TEMP/uhmap_manifest.jsonc", "r") as f:
        manifest = commentjson.load(f)

    uhmap_url = manifest[key]
    print('download main files | 下载预定文件')
    try:
        file = download(uhmap_url, filename="./TEMP/DOWNLOAD", unzip=True, unzip_path='./TEMP/UNZIP')
    except:
        print(f'download timeout | 下载失败, 您可能需要翻墙才能下载资源。另外如果您想手动下载的话: {uhmap_url}')
    return file

download_from_shared_server('EnvDesignTutorial') 
distutils.dir_util.copy_tree('./TEMP/UNZIP', './')

download_from_shared_server('uhmp-big-file-v3.1') 
distutils.dir_util.copy_tree('./TEMP/UNZIP', './')
