import os, commentjson, shutil, subprocess, tqdm, shutil, distutils
from onedrivedownloader import download

try: os.makedirs('./TEMP')
except: pass

def download_from_shared_server(key = 'cat'):
    # download uhmap file manifest | 下载manifest目录文件
    print('download uhmap file manifest | 下载manifest目录文件')
    manifest_url = "https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/EVmCQMSUWV5MgREWaxiz_GoBalBRV3DWBU3ToSJ5OTQaLQ?e=I8yjl9"

    try:
        file = download(manifest_url, filename="./TEMP/", force_download=True)
    except:
        print('failed to connect to onedrive | 连接onedrive失败, 您可能需要翻墙才能下载资源')

    with open("./TEMP/uhmap_manifest.jsonc", "r") as f:
        manifest = commentjson.load(f)
    if key not in key: 
        print('The version you are looking for does not exists!')
    uhmap_url = manifest[key]
    print('download main files | 下载预定文件')
    try:
        file = download(uhmap_url, filename="./TEMP/DOWNLOAD", unzip=True, unzip_path='./TEMP/UNZIP')
    except:
        print(f'download timeout | 下载失败, 您可能需要翻墙才能下载资源。另外如果您想手动下载的话: {uhmap_url}')
    return file

def download_client_binary_on_platform(desired_path, desired_version, is_render_client, platform):
    key = f"Uhmap_{platform}_Build_Version{desired_version}"
    print('downloading', key)
    download_from_shared_server(key = key)
    print('download and extract complete, moving files')
    from distutils import dir_util
    target_dir = os.path.abspath(os.path.dirname(desired_path) + './..')
    dir_util.copy_tree('./TEMP/UNZIP', target_dir)
    assert os.path.exists(desired_path), "unexpected path error! Are you using Linux style path on Windows?"
    return


def download_client_binary(desired_path, desired_version, is_render_client):
    import platform
    plat = "Windows"
    if platform.system()=="Linux": plat = "Linux"
    download_client_binary_on_platform(desired_path, desired_version, is_render_client, platform=plat)
    return

