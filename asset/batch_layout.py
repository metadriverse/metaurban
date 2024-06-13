import os
import json
from asset.read_config import configReader
from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.component.static_object.test_new_object import TestObject, TestGLTFObject
class batchLayoutStatic():
    def __init__(self, adj_folder):
        self.env_config = {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "debug": False,
            "cull_scene": False,
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "need_inverse_traffic": False,
            "rgb_clip": True,
            "map": "X",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": False,
            "window_size": (2400, 1600),
            "vehicle_config": {
                "enable_reverse": False,
            },
        }
        self.env = metaurbanEnv(config=self.env_config)
        o, _ = self.env.reset()
        self.adj_folder = adj_folder
        self.init_static_adj_list()
        # self.init_car_adj_list()
    def init_static_adj_list(self):
        self.static_asset_metainfos = []  # List to store the file paths
        for root, dirs, files in os.walk(self.adj_folder):
            for file in files:
                if not file.lower().startswith("car"):
                    print(file)
                    with open(os.path.join(root, file), 'r') as file:
                        loaded_metainfo = json.load(file)
                        print(loaded_metainfo)
                        self.static_asset_metainfos.append(loaded_metainfo)
    def init_car_adj_list(self):
        self.car_asset_metainfos = []  # List to store the file paths
        for root, dirs, files in os.walk(self.adj_folder):
            for file in files:
                if file.lower().startswith("car"):
                    print(file)
                    with open(os.path.join(root, file), 'r') as file:
                        loaded_metainfo = json.load(file)
                        print(loaded_metainfo)
                        self.car_asset_metainfos.append(loaded_metainfo)

    def spawn_static(self, lower_bound, upper_bound, block_size, lower_bound_x):
        num_positions_y = (upper_bound - lower_bound) // block_size
        grid_x = 0
        grid_y = 0

        for static_metainfo in self.static_asset_metainfos:
            # Compute real-world coordinates based on grid coordinates
            y_position = lower_bound + (grid_y * block_size)  # interchanged x with y here
            x_position = lower_bound_x + (grid_x * block_size)  # set a lower bound for x

            # Assign coordinates to object
            self.env.engine.spawn_object(TestObject, position=[x_position, y_position], heading_theta=0, random_seed=1,
                                         force_spawn=True,
                                         asset_metainfo=static_metainfo)

            # Update grid coordinates for next object
            grid_y += 1  # interchanged x with y here
            if grid_y >= num_positions_y:
                grid_y = 0  # Reset y position  # interchanged x with y here
                grid_x += 1  # Move to next column  # interchanged x with y here

    def step_env(self):
        while True:
            self.env.step([0, 0])


if __name__ == "__main__":
    config = configReader()
    path_config = config.loadPath()
    layout_helper = batchLayoutStatic(adj_folder=path_config["adj_parameter_folder"])
    layout_helper.spawn_static(-10, 10, 5, 10)
    layout_helper.step_env()



