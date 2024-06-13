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
 
# fqueue = open('queue.txt','a+')
# current_folder_id = folder_queue.pop(0)
# file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()
# # print(len(file_list)) #796
# print(file_list[0]['id'], file_list[0]['title'])
# for file in file_list:
#     fqueue.write(file['title'] + ' ' + file['id'] +'\n')
# fqueue.close()

fqueues = open('queue.txt','r').readlines()
f = open('script.sh', 'a+')   
visit = 1
for line in fqueues:
    current_fname, current_folder_id  = line.replace('\n','').split(' ')

    if os.path.exists('unconverted/'+current_fname.lower().replace('_fbx','_yup_t.fbx')): continue
    print(current_folder_id, current_fname)

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()

    # if cnt % 500 == 0 and cnt>0: time.sleep(3600)

    if cnt % 10 == 0:
        f.close()
        f = open('script.sh', 'a+')   
    for file in file_list:
        if '_yup_t.fbx' not in file['title']:  continue
        # print(file['id'] + ' unconverted/' + current_fname + '/' +  file['title'] + ' '+ file['md5Checksum'] )
        f.write(file['id'] + ' unconverted/' + current_fname + '/' +  file['title'] + ' '+ file['md5Checksum'] + '\n')
        bash_str = f"gdown --id {file['id']} -O unconverted/{file['title']}"
        os.system(bash_str)
       

    cnt += 1
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