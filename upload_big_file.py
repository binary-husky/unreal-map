import os, time
import commentjson as json
from onedrive_util import OneDrive
from datetime import datetime

# https://pypi.org/project/onedrive-sharepoint/
def get_onedrive_handle():
    # 第一步获取OneDrive访问句柄
    email = "fuqingxu@yiteam.tech"
    password = input('password please?')
    endpoint = "https://ageasga-my.sharepoint.com/personal/fuqingxu_yiteam_tech"
    type = "onedrive"
    session = OneDrive(email=email, password=password, endpoint=endpoint, type=type)
    return session


def add_file_to_onedrive(session, key, path_file_name_local):
    # 读取manifest目录
    dir_name = f"./{ datetime.now().strftime('%d-%m-%Y') }-datas"
    session.download_file('/personal/fuqingxu_yiteam_tech/Documents/ShareServer/uhmap_manifest.jsonc')
    manifest_path_local = os.path.relpath( os.path.join(dir_name, 'uhmap_manifest.jsonc') ).replace('\\','/')
    with open(manifest_path_local, 'r', encoding='utf8') as f:
        manifest = json.load(f)

    # 上传文件
    shareserver_remote_root = 'ShareServer/'
    session.upload_file_on_folder(path_file_name_local, shareserver_remote_root)
    # 创建公开链接
    share_link = session.share_folder(os.path.join(shareserver_remote_root, os.path.basename(path_file_name_local)), is_edit=False)
    manifest[key] = share_link

    with open(manifest_path_local, 'w', encoding='utf8') as f:
        json.dump(manifest, f, indent=4)

    # 上传manifest目录
    session.upload_file_on_folder(manifest_path_local, shareserver_remote_root)

    print('success')


# add_file_to_onedrive(
#     session = get_onedrive_handle(), 
#     key = 'uhmp-big-file-v3.1', 
#     path_file_name_local = 'PrivateUpload/uhmp-big-file-v3.1.zip')


add_file_to_onedrive(
    session = get_onedrive_handle(), 
    key = 'EnvDesignTutorial', 
    path_file_name_local = 'EnvDesignTutorial.zip')
