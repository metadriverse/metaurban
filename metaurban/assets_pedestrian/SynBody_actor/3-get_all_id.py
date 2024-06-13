

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
 
# 设置需要下载的Google Drive文件夹ID
parent_folder_id = '1SUakdglXmkS-KRsV2cxaora0JSQrGUFa'
# parent_folder_id = '1JrRCDe6GRLFaFRJaB4n1RGAeBtm_THAz' # SynBody/converted

file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(parent_folder_id)}).GetList()
print(file_list[0]['id'], file_list[0]['title'], int(file_list[0]['fileSize'])/1000000)

f = open('../RP_actor/converted_id.txt','a+')
for file in file_list:
    f.write(f"{file['id']}  {file['title']}  {int(file['fileSize'])/1000000}\n")
f.close()