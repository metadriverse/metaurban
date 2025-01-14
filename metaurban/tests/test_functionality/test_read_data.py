import os.path
from distutils.dir_util import copy_tree

from metaurban.engine.asset_loader import AssetLoader
from metaurban.envs.scenario_env import ScenarioEnv
from metaurban.scenario.scenario_description import ScenarioDescription
from metaurban.scenario.utils import read_dataset_summary, read_scenario_data


def test_read_waymo_data():
    summary_dict, summary_list, mapping = read_dataset_summary(AssetLoader.file_path("waymo", unix_style=False))
    for p in summary_list:
        data = read_scenario_data(AssetLoader.file_path("waymo", mapping[p], p, unix_style=False))
        data.sanity_check(data, check_self_type=False, valid_check=False)
        print("Finish: ", p)


def test_read_nuscenes_data():
    summary_dict, summary_list, mapping = read_dataset_summary(AssetLoader.file_path("nuscenes", unix_style=False))
    for p in summary_list:
        data = read_scenario_data(AssetLoader.file_path("nuscenes", mapping[p], p, unix_style=False))
        data.sanity_check(data, check_self_type=False, valid_check=False)
        print("Finish: ", p)
