"""
This script contains the configReader class, which is designed to manage and read configuration settings
from YAML files. The class handles the loading of various configurations, including paths, types, tags,
and other relevant data needed for asset management and processing.
It facilitates easy access to these configurations throughout the project.

The configReader class offers methods to load and update configurations,
making it a central point for handling all configuration-related tasks in the project.

Class Methods:
- __init__: Initializes the configReader instance, loading path and asset configuration from YAML files.
- loadSubPath: Constructs full file paths from parent and child folder paths.
- loadPath: Loads various file paths from the path configuration.
- loadTag: Retrieves tag configuration.
- loadType: Retrieves type configuration.
- loadTypeInfo: Loads detailed information for each type from the asset configuration.
- loadColorList: Retrieves a list of colors from the asset configuration.
- loadCarType: Retrieves the types of vehicles from the type configuration.
- getReverseType: Creates a reverse mapping from detailed to general types.
- getSpawnNum: Retrieves spawn number for a given detailed type.
- getSpawnPos: Retrieves spawn position for a given detailed type.
- getSpawnHeading: Retrieves spawn heading for a given detailed type.
- updateTypeInfo: Updates type information in the asset configuration.
"""
import yaml
import os
from pathlib import Path
from typing import Dict
class configReader:
    def __init__(self, config_path = "path_config.yaml"):
        """
        Initializes the configReader instance, loading path and asset configuration from YAML files.

        Parameters:
        - config_path (str): Path to the YAML configuration file.

        Returns:
        - None
        """
        self.spawnPosDict = None
        self.spawnNumDict = None
        self.reverseType = None
        self.path_config_path = "./path_config.yaml"
        self.asset_config_path = "./asset_config.yaml"
        with open(self.path_config_path, "r") as file:
            self.path_config = yaml.safe_load(file)
        with open(self.asset_config_path, "r") as file2:
            self.asset_config = yaml.safe_load(file2)
    def loadSubPath(self, parent_folder: str, child_folder_dict: Dict):
        """
        Constructs full file paths by combining a parent folder with relative child folder paths.

        Parameters:
        - parent_folder (str): Path to the parent folder.
        - child_folder_dict (Dict): Dictionary of child folder names and their relative paths.

        Returns:
        - Dict: Dictionary with updated full paths for each child folder.
        """
        unified_parent_folder = Path(parent_folder)
        result_folder_dict = {}
        for key, path in child_folder_dict.items():
            result_folder_dict[key] = os.path.join(unified_parent_folder, Path(path))
        return result_folder_dict

    def loadPath(self):
        """
        Loads and returns a dictionary of various file paths from the path configuration.

        Returns:
        - Dict: Dictionary containing various file paths.
        """
        result_folder_dict = {}
        path_dict = self.path_config['path']
        for key, path in path_dict.items():
            if key == "subfolders":
                concat_path_dict = self.loadSubPath(parent_folder = path_dict["parentfolder"],
                                                 child_folder_dict = path
                                                 )
                result_folder_dict.update(concat_path_dict)
            else:
                result_folder_dict[key] = path
        return result_folder_dict
    def loadTag(self):
        """
        Retrieves and returns the tag configuration.

        Returns:
        - Dict: Dictionary containing tag configurations.
        """
        return self.asset_config["tag"]
    def loadType(self):
        """
        Retrieves and returns the type configuration.

        Returns:
        - Dict: Dictionary containing type configurations.
        """
        return self.asset_config["type"]
    def loadTypeInfo(self):
        """
        Loads and returns detailed information about each type from the asset configuration.

        Returns:
        - Dict: Dictionary containing detailed information for each type.
        """
        with open(self.asset_config_path, "r") as file:
            self.asset_config = yaml.safe_load(file)
        return self.asset_config["typeinfo"]
    def loadColorList(self):
        """
        Retrieves and returns a list of colors from the asset configuration.

        Returns:
        - list[str]: List of color names.
        """
        return self.asset_config["others"]["color"]
    def loadCarType(self):
        """
        Retrieves and returns the keys (types) of vehicles from the type configuration.

        Returns:
        - list[str]: List of vehicle types.
        """
        return self.asset_config["type"]["vehicle"].keys()
    def getReverseType(self):
        """
        Creates and stores a reverse mapping from detailed types to general types in the asset configuration.

        Returns:
        - None
        """
        self.reverseType = dict()
        for general_type, detail_type_dict in self.asset_config['type'].items():
            for detail_type in detail_type_dict.keys():
                self.reverseType[detail_type] = general_type
    def getSpawnNum(self, detail_type):
        """
        Retrieves the spawn number for a given detailed type from the asset configuration.

        Parameters:
        - detail_type (str): The detailed type to retrieve the spawn number for.

        Returns:
        - int: The spawn number for the specified type.
        """
        if self.reverseType is None:
            self.getReverseType()
        return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawnnum"]
    
    def getSpawnInterval(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        try:
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawn_long_gap"]
        except:
            return 0
        
    def getSpawnLatInterval(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        try:
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawn_lat_gap"]
        except:
            return 0
        
    def getrandom_gap(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        try:
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["random_gap"]
        except:
            return False
        
    def get_rank(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        try:
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["rank_of_the_type"]
        except:
            return 123456
    
    def getSpawnPos(self, detail_type):
        """
        Retrieves the spawn position for a given detailed type from the asset configuration.

        Parameters:
        - detail_type (str): The detailed type to retrieve the spawn position for.

        Returns:
        - list[float]: The spawn position for the specified type.
        """
        if self.reverseType is None:
            self.getReverseType()
        return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawnpos"]
    def getSpawnHeading(self, detail_type):
        """
        Retrieves the spawn heading for a given detailed type from the asset configuration.

        Parameters:
        - detail_type (str): The detailed type to retrieve the spawn heading for.

        Returns:
        - float or bool: The spawn heading for the specified type, or False if not defined.
        """
        if self.reverseType is None:
            self.getReverseType()
        if "spawnheading" in self.asset_config['type'][self.reverseType[detail_type]][detail_type].keys():
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["spawnheading"]
        return False
    def updateTypeInfo(self, new_info_dict):
        """
        Updates the type information in the asset configuration with new information.

        Parameters:
        - new_info_dict (Dict): Dictionary containing new type information to update.

        Returns:
        - None
        """
        for key, val in new_info_dict.items():
            self.asset_config["typeinfo"][key] = val
            with open(self.asset_config_path, "w") as file:
                yaml.safe_dump(self.asset_config, file)
