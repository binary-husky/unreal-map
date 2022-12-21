import os, commentjson, shutil, subprocess, tqdm, shutil
from onedrivedownloader import download
from distutils import dir_util

try: os.makedirs('./TEMP')
except: pass

def download_from_shared_server(key = 'cat'):
    # download uhmap file manifest | 下载manifest目录文件
    print('download uhmap file manifest | 下载manifest目录文件')
    manifest_url = "https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/EVmCQMSUWV5MgREWaxiz_GoBalBRV3DWBU3ToSJ5OTQaLQ?e=I8yjl9"

    try:
        download(manifest_url, filename="./TEMP/", force_download=True)
    except:
        print('failed to connect to onedrive | 连接onedrive失败, 您可能需要翻墙才能下载资源')
        return False

    with open("./TEMP/uhmap_manifest.jsonc", "r") as f:
        manifest = commentjson.load(f)

    uhmap_url = manifest[key]
    print('download main files | 下载预定文件')
    try:
        download(uhmap_url, filename="./TEMP/DOWNLOAD", unzip=True, unzip_path='./TEMP/UNZIP')
    except:
        print(f'download timeout | 下载失败, 您可能需要翻墙才能下载资源。另外如果您想手动下载的话: {uhmap_url}')
        return False

    return True

def get_current_version():
    with open('current_version', 'r', encoding='utf8') as f:
        version = f.read()
        return version

version = get_current_version()
success = download_from_shared_server(f'uhmp-big-file-v{version}') 

if success:
    print('Copying downloaded files to project root')
    print('This will take a while if you are not using SSD...')
    dir_util.copy_tree('./TEMP/UNZIP', './')
    print('下载完成')
else:
    try: shutil.rmtree('./TEMP')
    except: pass
    print('下载失败, 临时文件夹已清除')
