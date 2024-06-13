from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os, sys 
import time 
import pickle 
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
 
 
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
 
# 设置需要下载的Google Drive文件夹ID
parent_folder_id = '1OK8iD2idSHbdU0KKf-coUpruKBbP0PkA'
 
# 设置你在Bash上的下载路径。
parent_folder_dir = 'unconverted'
 
if parent_folder_dir[-1] != '/':
  parent_folder_dir = parent_folder_dir + '/'
 
# 使用gdown进行下载
wget_text = '"[ ! -f FILE_NAME ] && gdown --no-cookies https://drive.google.com/uc?id=FILE_ID -O FILE_NAME"'
md5_text = '"MD5_SUM FILE_NAME"'
 
# Get the folder structure
 
file_dict = dict()
folder_queue = [parent_folder_id]
dir_queue = [parent_folder_dir]
cnt = 0
 
f = open('script.sh', 'a+')   
# fmd5 = open('script.md5', 'a+')  
visit = 1
while len(folder_queue) != 0:
    
    if visit % 500 == 0: time.sleep(3600)

    current_folder_id = folder_queue.pop(0)
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()
    # print(len(file_list)) #796
    print(file_list[0])
    assert False

    current_parent = dir_queue.pop(0)
    print('--', current_parent, current_folder_id)
    for file1 in file_list:
        if cnt % 10 == 0:
            f.close()
            f = open('script.sh', 'a+')   

        # print('file1: ', file1)
        file_dict[cnt] = dict()
        file_dict[cnt]['id'] = file1['id']
        file_dict[cnt]['title'] = file1['title']
        file_dict[cnt]['dir'] = current_parent + file1['title']

        if file1['mimeType'] == 'application/vnd.google-apps.folder':
            if '/tex/' in file1['title']:  continue  
            file_dict[cnt]['type'] = 'folder'
            file_dict[cnt]['dir'] += '/'
            folder_queue.append(file1['id'])
            dir_queue.append(file_dict[cnt]['dir'])
        else: # for file
            if '_yup_t.fbx' not in file1['title']: 
                continue
            file_dict[cnt]['type'] = 'file'
            file_dict[cnt]['md5Checksum'] = file1['md5Checksum']
            bash_str = f"gdown --id {file1['id']} -O unconverted/{file1['title']}"
            os.system(bash_str)
            f.write(file1['id'] + ' ' + current_parent + file1['title'] + ' '+ file1['md5Checksum'] + '\n')

        cnt += 1
    visit += 1
f.close()
assert False

f = open('script.sh', 'w')   
fmd5 = open('script.md5', 'w')  
print('file_dict.keys(): ', len(file_dict.keys()))
for file_iter in file_dict.keys():

    # if file_dict[file_iter]['type'] == 'folder' and '/tex/' not in file_dict[file_iter]['title']:
    #     f.write('mkdir ' + file_dict[file_iter]['dir'] + '\n')
    #     os.makedirs(file_dict[file_iter]['dir'], exist_ok=True)

    if file_dict[file_iter]['type'] == 'file' and '_yup_t.fbx' in file_dict[file_iter]['title']:
        print(file_dict[file_iter])
        bash_str = f"gdown --id {file_dict[file_iter]['id']} -O unconverted/{file_dict[file_iter]['title']}"
        os.system(bash_str)
        f.write(file_dict[file_iter]['id'] + ' ' + file_dict[file_iter]['dir'] + '\n')
        fmd5.write(md5_text[1:-1].replace('MD5_SUM', file_dict[file_iter]['md5Checksum']).replace('FILE_NAME', file_dict[file_iter]['dir']) + '\n')

    # if file_dict[file_iter]['type'] == 'folder':
    #     f.write('mkdir ' + file_dict[file_iter]['dir'] + '\n')
    #     os.makedirs(file_dict[file_iter]['dir'], exist_ok=True)
    # else:
    #     bash_str = wget_text[1:-1].replace('FILE_ID', file_dict[file_iter]['id']).replace('FILE_NAME', file_dict[file_iter]['dir'])
    #     os.system(bash_str)
    #     f.write(bash_str + '\n')
    #     fmd5.write(md5_text[1:-1].replace('MD5_SUM', file_dict[file_iter]['md5Checksum']).replace('FILE_NAME', file_dict[file_iter]['dir']) + '\n')

f.close()
assert False



# import gdown
# import requests

# # url = 'https://drive.google.com/uc?id=1s7mgGTYn6-8ZIIajPU6ldBs36XmW'
# # https://drive.google.com/drive/folders/1OK8iD2idSHbdU0KKf-coUpruKBbP0PkA?usp=sharing

# def list_files(folder_id):
#     url = f"https://drive.google.com/drive/v3/files?q='{folder_id}'+in+parents&fields=files(name)"
#     response = requests.get(url)
#     print(response)
#     if response.status_code == 200:
#         files = response.json().get('files',[])
#         for file in files:
#             print(file.get('name'))

# list_files('1OK8iD2idSHbdU0KKf-coUpruKBbP0PkA')
# # gdown.extractall(url, 'test/')

# https://drive.google.com/drive/folders/1OK8iD2idSHbdU0KKf-coUpruKBbP0PkA?usp=sharing


# subfolder:
# https://drive.google.com/drive/folders/1s7mgGTYn6-8ZIIajPU6ldBs36XmW-urx?usp=sharing