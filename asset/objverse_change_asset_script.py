import json
import os
import shutil
import yaml
from pathlib import Path
from asset.objverse_change_asset import AssetMetaInfoUpdater
from asset.objverse_change_asset_static import StaticAssetMetaInfoUpdater
from asset.objverse_autochange_asset_static import AutoStaticAssetMetaInfoUpdater
from asset.objverse_autochange_asset import AutoAssetMetaInfoUpdater
from asset.read_config import configReader
def load_json(file_path):
    """
    Load json file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def copy_file(uid, src_folder, src, dst_folder, tag):
    """
    Copy the source file to the destination folder and rename it using the uid.
    Returns the path to the copied file.
    """
    true_filename = os.path.basename(src)
    dst = f"{tag}-{uid}.glb"
    dst_folder, dst = Path(dst_folder), Path(dst)
    src_folder, src = Path(src_folder), Path(src)
    return_path = os.path.join(dst_folder, dst)
    shutil.copy(os.path.join(src_folder,src), return_path)
    return return_path
def delete_file(filepath):
    """
    Delete a file given its filepath.
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print(f"Error: {filepath} not found!")

def save_ignore_list(file_path, ignore_list):
    """Save the ignore list to a file."""
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w') as file:
        json.dump(ignore_list, file)
def delete_folder(folder_path):
    """
    Delete a folder and all its contents.

    :param folder_path: Path to the folder to be deleted.
    :return: None
    """
    # Check if the folder exists before attempting to delete
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder '{folder_path}' does not exist!")
def model_update(is_auto, is_car_model, destination_folder, json_path, adj_parameter_folder, src_parent_folder, ignore_list_path):
    print(destination_folder)
    if os.path.exists(ignore_list_path):
        ignore_list = load_json(ignore_list_path)
    else:
        ignore_list = []
    data = load_json(json_path)
    for uid, relative_path in data.items():
        if uid in ignore_list:
            print(f"UID {uid} is in the ignore list. Skipping...")
            continue
        print("dealing with: {}".format(uid))
        # First copy the model into metaurban's model folder
        copied_model_path = copy_file(uid, src_parent_folder, relative_path,
                                      destination_folder, tag=tag)  # This will now be your new model_path_input
        save_path = os.path.join(adj_parameter_folder, f"{tag}-{uid}.json")
        if is_auto:
            if is_car_model:
                updater = AutoAssetMetaInfoUpdater("test/" + os.path.basename(copied_model_path), adj_parameter_folder, uid)
            else:
                updater = AutoStaticAssetMetaInfoUpdater(os.path.basename(copied_model_path), adj_parameter_folder, uid)
        else:
            if is_car_model:
                updater = AssetMetaInfoUpdater("test/" + os.path.basename(copied_model_path), save_path)
            else:
                updater = StaticAssetMetaInfoUpdater(os.path.basename(copied_model_path), save_path)
        save_flag = updater.run()
        if not save_flag:
            delete_file(copied_model_path)
            ignore_list.append(uid)
            print(ignore_list)
        save_ignore_list(ignore_list_path, ignore_list)
def folder_asset_update(is_auto, is_car_model, destination_folder, adj_parameter_folder, raw_asset_src_folder, ignore_list_path):
    if os.path.exists(ignore_list_path):
        ignore_list = load_json(ignore_list_path)
    else:
        ignore_list = []
    for filename in os.listdir(raw_asset_src_folder):
        try:
            tag, rest = filename.split('-')
            uid, fileextension = rest.split('.')
        except ValueError:
            print("Wrong asset name format, should be tag-uid.glb/gltf")
        if uid in ignore_list:
            print(f"UID {uid} is in the ignore list. Skipping...")
            continue
        print("dealing with: {}".format(uid))
        # First copy the model into metaurban's model folder
        copied_model_path = copy_file(uid, raw_asset_src_folder, filename,
                                      destination_folder, tag=tag)  # This will now be your new model_path_input
        # save_path = os.path.join(adj_parameter_folder, f"{tag}-{uid}.json")
        if is_auto:
            updater = AutoStaticAssetMetaInfoUpdater(os.path.basename(copied_model_path), adj_parameter_folder, uid)
        elif is_car_model:
            updater = AssetMetaInfoUpdater("test/" + os.path.basename(copied_model_path), save_path)
        else:
            updater = StaticAssetMetaInfoUpdater(os.path.basename(copied_model_path), save_path)
        save_flag = updater.run()
        if not save_flag:
            delete_file(copied_model_path)
            ignore_list.append(uid)
            print(ignore_list)
        save_ignore_list(ignore_list_path, ignore_list)
def gltf_updater(destination_folder, adj_parameter_folder, src_parent_folder, ignore_list_path):
    if os.path.exists(ignore_list_path):
        ignore_list = load_json(ignore_list_path)
    else:
        ignore_list = []
    for subfolder in os.listdir(src_parent_folder):
        subfolder_path = os.path.join(src_parent_folder, subfolder)
        if subfolder in ignore_list:
            print(f"Ignore folder {subfolder}")
        # Check if it's actually a directory and not a file
        if os.path.isdir(subfolder_path):
            gltf_file_path = os.path.join(subfolder_path, 'scene.gltf')
            # Check if the 'scene.gltf' file exists in this subfolder
            if os.path.exists(gltf_file_path):
                print(f"Found 'scene.gltf' in {subfolder}")
                save_path = os.path.join(adj_parameter_folder, f"{tag}-{subfolder}.json")
                dest_subfolder_path = os.path.join(destination_folder, subfolder)
                if not os.path.exists(dest_subfolder_path):
                    shutil.copytree(subfolder_path, dest_subfolder_path)
                    print(f"Copied {subfolder} to {dest_subfolder_path}")
                updater = StaticAssetMetaInfoUpdater("scene.gltf", save_path, folder_name=subfolder, isGLTF=True)
            else:
                print(f"No 'scene.gltf' found in {subfolder}")
            save_flag = updater.run()
            if not save_flag:
                delete_folder(dest_subfolder_path)
                ignore_list.append(subfolder)
                print(ignore_list)
            save_ignore_list(ignore_list_path, ignore_list)


if __name__ == "__main__":
    config = configReader()

    tag_config = config.loadTag()
    tag = tag_config["tag"]
    istaglist = tag_config["istaglist"]

    path_config = config.loadPath()
    asset_folder_path = path_config["assetfolder"]

    raw_asset_path_folder = path_config["raw_asset_path_folder"]
    filter_asset_path_folder = path_config["filter_asset_path_folder"]
    match_uid_path_folder = path_config["match_uid_path_folder"]
    # Folder where you want to copy the asset into, should be this path, otherwise metaurban won't recognize it.
    destination_folder = path_config["metaurbanasset"]  # Folder where you want to copy the files
    # Assets you want to use, with their paths. Generated from objverse_filter_asset.py
    match_uid_path_folder = path_config["match_uid_path_folder"]
    json_path = os.path.join(match_uid_path_folder, "matched_uids_{}.json".format(tag))
    # Folder where you want to save Adjusted parameters to
    adj_parameter_folder = path_config["adj_parameter_folder"]
    # Original asset parent folder (the path from above matched.json is relative)
    src_parent_folder = path_config["assetfolder"]
    # List of uids you want to ignore.
    ignore_adj_folder = os.path.join(path_config["ignore_adj_folder"], "ignore_list_{}.json".format(tag))
    # ===========================================Car Model=======================================
    # model_update(is_auto=True,
    #              is_car_model=True,
    #              destination_folder= destination_folder,
    #              json_path = json_path,
    #              adj_parameter_folder = adj_parameter_folder,
    #              src_parent_folder = src_parent_folder,
    #              ignore_list_path = ignore_adj_folder)
    # ===========================================Static Model=======================================
    model_update(is_auto = True,
                 is_car_model=False,
                 destination_folder= destination_folder,
                 json_path = json_path,
                 adj_parameter_folder= adj_parameter_folder,
                 src_parent_folder = src_parent_folder,
                 ignore_list_path = ignore_adj_folder)
    # ===========================================GLTF Model=======================================
    # src_parent_folder = path_config["metaurbanassetgltf"]  # Folder where you want to copy the files
    # gltf_updater(destination_folder = destination_folder,
    #              adj_parameter_folder= adj_parameter_folder,
    #              src_parent_folder = src_parent_folder,
    #              ignore_list_path = ignore_adj_folder)
    # ============================================Raw Asset Folder==================================
    # raw_asset_src_folder = path_config["raw_assetfolder"]
    # folder_asset_update( is_auto = True,
    #             is_car_model=False,
    #             destination_folder = destination_folder,
    #             adj_parameter_folder= adj_parameter_folder,
    #             raw_asset_src_folder = raw_asset_src_folder,
    #             ignore_list_path = ignore_adj_folder)