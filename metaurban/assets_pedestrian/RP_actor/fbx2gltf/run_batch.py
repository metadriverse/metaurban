import os
folder_path = ""
paths = [os.path.abspath(os.path.join(folder_path, x)) for x in os.listdir(folder_path)]

from glob import glob
for fbxpath in paths:
    
    if os.path.exists(fbxpath.replace('/unconverted/','/converted/')+'.gltf'): continue
    output_path = fbxpath.replace('/unconverted/','/converted/')
    # print(f"{path}/*_yup_t.fbx")
    # fbxpath = glob(f"{path}/*_yup_t.fbx")[0]
    os.system(f"bash run.sh {fbxpath} {output_path}")
