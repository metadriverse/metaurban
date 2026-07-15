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
- getReverseType: Creates a reverse mapping from detailed to general types.
- getSpawnNum / getSpawnInterval / getrandom_gap: Per-detail-type spawn policy.
"""
import yaml
import os
from pathlib import Path
from typing import Dict
script_dir = os.path.dirname(os.path.abspath(__file__))

class configReader:
    def __init__(self, config_path="path_config.yaml"):
        """
        Initializes the configReader instance, loading path and asset configuration from YAML files.

        Parameters:
        - config_path (str): Path to the YAML configuration file.

        Returns:
        - None
        """
        self.reverseType = None
        self.path_config_path = f"{script_dir}/../../path_config.yaml"
        self.asset_config_path = f"{script_dir}/../../asset_config.yaml"
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
                concat_path_dict = self.loadSubPath(parent_folder=f"{script_dir}/../../" + path_dict["parentfolder"], child_folder_dict=path)
                result_folder_dict.update(concat_path_dict)
            else:
                result_folder_dict[key] = f"{script_dir}/../../" + path
        return result_folder_dict

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

    def getrandom_gap(self, detail_type):
        if self.reverseType is None:
            self.getReverseType()
        try:
            return self.asset_config['type'][self.reverseType[detail_type]][detail_type]["random_gap"]
        except:
            return False
