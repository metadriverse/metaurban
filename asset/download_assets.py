# Download Asset with given Tag or Tag List from Objaverse
import objaverse
import json
import os
from tqdm import tqdm
import yaml
from asset.read_config import configReader

class Objverse_helper:
    def __init__(self):
        pass
    def getAllTag(self, num_uids=-1):
        """
        Retrieves all unique tags from the Objaverse with the given number of uids.

        Parameters:
        - num_uids (int): The number of UIDs to retrieve. If -1, retrieves all UIDs.

        Returns:
        - annotations: Objaverse annotation dict with key as uid and val as content
        - tag: Unique tags from the given uids
        """
        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        tag = set()
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                tag.add(eachtag["name"])
        return annotations, tag
    def getTagStrictly(self, num_uids=-1, target_tag = "car", strict=True):
        """
        Searches annotations to retrieve specified tags. Can search strictly or loosely based on the target tag.

        Parameters:
        - num_uids (int): The number of UIDs to retrieve. If -1, retrieves all UIDs.
        - target_tag (str): The target tag to search for.
        - strict (bool): If True, searches for exact matches. If False, searches for partial matches.

        Returns:
        - uid_list (list): List of UIDs that matched the criteria.
        - full_list (list): List of annotations corresponding to the matched UIDs.
        - full_tag (list): List of all the tags found.
        """

        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        full_tag = []
        uid_list = []
        full_list = []
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                full_tag.append(eachtag['name'])
                if strict:
                    if eachtag['name'] == target_tag:
                        uid_list.append(uid)
                        full_list.append(content)
                else:
                    if target_tag in eachtag["name"]:
                        uid_list.append(uid)
                        full_list.append(content)
        return uid_list, full_list, full_tag
    def getTagList(self, num_uids=-1, target_tag_list = []):
        """
        Searches annotations to retrieve any tags that are in the provided target tag list.

        Parameters:
        - num_uids (int): The number of UIDs to retrieve. If -1, retrieves all UIDs.
        - target_tag_list (list): List of target tags to search for.

        Returns:
        - uid_list (list): List of UIDs that matched the criteria.
        - full_list (list): List of annotations corresponding to the matched UIDs.
        - full_tag (list): List of all the tags found.
        """
        uids = objaverse.load_uids()
        print("Load all annotations...")
        annotations = objaverse.load_annotations(uids[:num_uids])
        full_tag = []
        uid_list = []
        full_list = []
        for uid, content in tqdm(annotations.items(), desc="Parsing all annotations"):
            for eachtag in content["tags"]:
                full_tag.append(eachtag['name'])
                if eachtag["name"] in target_tag_list:
                    uid_list.append(uid)
                    full_list.append(content)
        return uid_list, full_list, full_tag
    def saveTag(self, objects, tagname, parent_folder, isTagList = False, tagList = None):
        """
        Saves the provided objects path information to a JSON file with a specified name.

        Parameters:
        - objects (dict): The objects uid(key) and path(val) to save.
        - tagname (str): The name used for naming the saved JSON file.
        - parent_folder (str): The directory in which the file will be saved.
        - isTagList (bool): If True, a tag list will be included in the saved objects (for example, human might include man and woman tags).
        - tagList (list, optional): List of tags to be included in the saved objects if isTagList is True.

        Returns:
        - None
        """
        if isTagList:
            assert tagList is not None
        if not isTagList:
            path = "object-paths-{}.json".format(tagname)
        else:
            path = "object-list-paths-{}.json".format(tagname)
        with open(os.path.join(parent_folder, path), "w") as f:
            if isTagList:
                objects["tagList"] = tagList
            json.dump(objects, f)


if __name__ == "__main__":
    config = configReader()
    path_config = config.loadPath()
    processes = 1
    objhelper = Objverse_helper()
    #======================================Get All Tags from Objverse===============
    # _, tag = objhelper.getAllTag()
    # with open(path_config["all_tag_path"], "w+") as f:
    #     for each in tag:
    #         f.write("{}\n".format(each))

    # ======================================Get UIDS and objects for list of TAGs===============
    tag_config = config.loadTag()
    tag = tag_config["tag"]
    istaglist = tag_config["istaglist"]
    taglist = tag_config["taglist"]
    if istaglist:
        uid_list, full_list, full_tag = objhelper.getTagList(num_uids=-1, target_tag_list = taglist)
        uid_list.sort()
        print(len(uid_list))
    else:
        uid_list, full_list, full_tag = objhelper.getTagStrictly(-1, tag, strict=False)
        print(len(uid_list))
    #============================Note: This will download the asset and return, could be time consuming
    #============================Note: Objects is a dict with key as uid, and val as path
    objects = objaverse.load_objects(
        uids=uid_list[:150],
        download_processes=processes
    )
    raw_asset_path_folder = path_config["raw_asset_path_folder"]
    if not os.path.exists(raw_asset_path_folder):
        os.mkdir(raw_asset_path_folder)
    objhelper.saveTag(objects,tagname=tag, parent_folder=raw_asset_path_folder, isTagList=istaglist, tagList=taglist)
