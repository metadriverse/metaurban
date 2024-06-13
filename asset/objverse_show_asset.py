import time
import objaverse
import os
import trimesh
import subprocess
import tkinter as tk
from tkinter import simpledialog, messagebox
import json

from metaurban.envs.show_asset_metaurban_env import ShowAssetmetaurbanEnv


class Objverse_show_asset:
    def __init__(self, asset_metainfo_json_path):
        with open(asset_metainfo_json_path, "r") as file:
            self.asset_metainfo = json.load(file)
            print(self.asset_metainfo)
        self.env_config = {
            "manual_control": True,
            "use_render": True,
            "window_size": (1600, 1100),
            "start_seed": 1000,
            "test_asset_meta_info": self.asset_metainfo
        }
        self.env = ShowAssetmetaurbanEnv(config=self.env_config)
        o, _ = self.env.reset()
        o, r, tm, tc, info = self.env.step(
                                           actions=[0, 0])
    def run(self):
        while True:
            o, r, tm, tc, info = self.env.step(actions=[0, 0])
if __name__ == "__main__":
    objaverse_show_helper = Objverse_show_asset("C:\\research\\gitplay\\MetaVQA\\asset\\1827be72ffd44b51870aecf67e6cca71.json")
    # objaverse_show_helper = Objverse_show_asset(
    #     "C:\\research\\gitplay\\MetaVQA\\asset\\35002d28d3fe4359be492bd444569744.json")
    objaverse_show_helper.run()