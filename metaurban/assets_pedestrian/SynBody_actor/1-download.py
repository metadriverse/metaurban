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

 
 
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
 

## unconverted: 
## converted: 1xAjHnEJ-kdfNZhV9yykTlB1GqMEW4oMK
# 设置需要下载的Google Drive文件夹ID
parent_folder_id = '1iLKoFV1qAy2CfCzHbWIUFQ_hNW1DuWCV'
 
 #
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
 
if not os.path.exists('queue.txt'):
    fqueue = open('queue.txt','a+')
    current_folder_id = folder_queue.pop(0)
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()
    # print(len(file_list)) #796
    print(file_list[0]['id'], file_list[0]['title'])
    for file in file_list:
        fqueue.write(file['title'] + ' ' + file['id'] +'\n')
    fqueue.close()

fqueues = open('queue.txt','r').readlines()
f = open('script_tex.sh', 'a+')   
visit = 1
for line in fqueues:
    current_fname, current_folder_id  = line.replace('\n','').split(' ')

    # if os.path.exists('unconverted/'+current_fname+'.gltf'): continue
    # if os.path.exists(f'unconverted/{current_fname}/scene.blend'): continue
    if os.path.exists(f'unconverted/{current_fname}/textures'): continue

    print(current_folder_id, current_fname)

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()

    # if cnt % 500 == 0 and cnt>0: time.sleep(3600)

    if cnt % 10 == 0:
        f.close()
        f = open('script_tex.sh', 'a+')   
    # print(file_list)
    # assert False
    for file in file_list:
        if 'textures' != file['title']: continue
        # if 'scene.blend' != file['title']: continue     # for unconverted
        # print(file)
        # assert False
        #if '.gltf' not in file['title']:  continue   #for converted

        print(file['id'] + ' unconverted/' + current_fname + '/' +  file['title']  ) #+ ' '+ file['md5Checksum']
        f.write(file['id'] + ' unconverted/' + current_fname + '/' +  file['title'] + '\n') #+ ' '+ file['md5Checksum']
        os.makedirs(f'unconverted/{current_fname}/textures', exist_ok=True)
        # bash_str = f"gdown --id {file['id']} -O unconverted/{current_fname}/{file['title']}"
        bash_str = f"gdown --id {file['id']} -O unconverted/{current_fname}/{file['title']} --folder"

        os.system(bash_str)
        # assert False

    cnt += 1
f.close()
assert False