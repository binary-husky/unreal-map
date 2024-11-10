import os, commentjson, shutil, subprocess, tqdm, shutil
import zipfile
from modelscope import snapshot_download
try: os.makedirs('./TEMP')
except: pass

version = 'unreal-map-v3.4'
model_dir = snapshot_download(f'BinaryHusky/{version}')
zip_file_path = f'./TEMP/{version}.zip'

def combine_file(model_dir, output_file_path, num_parts):
    with open(output_file_path, 'wb') as output_file:
        for i in range(0, num_parts):
            part_file_path = os.path.join(model_dir, "tensor", f"safetensor_{i+1}.pt")
            with open(part_file_path, 'rb') as part_file:
                output_file.write(part_file.read())

extract_to_path = './'
combine_file(model_dir, output_file_path=zip_file_path, num_parts=5)

# 打开 ZIP 文件
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # 解压所有文件到指定目录
    zip_ref.extractall(extract_to_path)
    print(f"files unzipped {extract_to_path}")

print("everything is ready!")
