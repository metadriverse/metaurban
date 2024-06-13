# Script used to Add static model\GLTF model (not car model) and make sure it has proper size and proper tire position.
# Please refer to "CustomizedCar" class in metaurban/component/vehicle/vehicle_type.py for how the car model class is defined
# You don't need to understand how ttk works, just use it!
# Use objverse_change_asset_script.py to call this updater for each newly added car asset.
# Please refer to "objverse_change_asset.py" for detailed comments
import tkinter as tk
from tkinter import ttk
from functools import partial
from metaurban.envs.metaurban_env import metaurbanEnv
import json
import os
from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.component.static_object.test_new_object import TestObject, TestGLTFObject
from metaurban.component.static_object.traffic_object import TrafficCone, TrafficWarning
class StaticAssetMetaInfoUpdater:
    def __init__(self, file_name, save_path=None, folder_name=None, isGLTF = False):
        self.isGLTF = isGLTF
        self.asset_metainfo = {
        "length": 2,
        "width": 2,
        "height": 2,
        "filename": file_name,
        "foldername": folder_name,
        "CLASS_NAME": file_name,
        "hshift": 0,
        "pos0": 0,
        "pos1": 0,
        "pos2": 0,
        "scale": 1
    }
        self.RANGE_SPECS = {
            "length": (0, 10),
            "width": (0, 10),
            "height": (0, 10),
            "hshift": (-180,180),
            "pos0": (-10, 10),
            "pos1": (-10, 10),
            "pos2": (-10, 10),
            "scale": (0.01, 20),
        }
        if save_path and os.path.exists(save_path):
            with open(save_path, 'r') as file:
                loaded_metainfo = json.load(file)
                self.asset_metainfo.update(loaded_metainfo)  # update the asset_metainfo with the values from the file
        self.env_config = {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "debug": False,
            "cull_scene": False,
            "manual_control": False,
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
        self.entries = {}
        self.env = metaurbanEnv(config=self.env_config)
        o, _ = self.env.reset()
        self.root = tk.Tk()
        self.root.title("Asset MetaInfo Updater")
        self.setup_ui()
        self.save_path = save_path
        self.run_result = None
        self.current_obj = None
        self.env.engine.spawn_object(TrafficWarning, position=[10, -5], heading_theta=0,
                                     random_seed=1)

    def slider_command(self, v, key, idx=None):
        self.update_value(key, v, idx)

    def update_value(self, name, value, index=None):
        """Updates the asset_metainfo with the new value."""
        new_val = float(value)
        self.asset_metainfo[name] = new_val

        # Update the corresponding StringVar in self.entries
        if name in self.entries:
            if isinstance(self.entries[name], dict):  # Check if it's a dictionary
                for idx, entry_var in self.entries[name].items():
                    entry_var.set(value)
            else:
                self.entries[name].set(value)

    def environment_step(self):
        """Steps through the environment with the updated asset_metainfo configuration and schedules itself to run again."""
        if self.current_obj is not None:
            self.env.engine.clear_objects([self.current_obj.id], force_destroy=True)
        if self.isGLTF:
            self.current_obj = self.env.engine.spawn_object(TestGLTFObject, position=[10, -5], heading_theta=0,
                                                            random_seed=1, force_spawn=True,
                                                            asset_metainfo=self.asset_metainfo)
        else:
            self.current_obj = self.env.engine.spawn_object(TestObject, position=[10, -5], heading_theta=0, random_seed=1, force_spawn=True,
                                    asset_metainfo=self.asset_metainfo)
        amin_point, amax_point = self.current_obj.origin.getTightBounds()
        p1 = amax_point[0], amax_point[1]
        p2 = amax_point[0], amin_point[1]
        p3 = amin_point[0], amin_point[1]
        p4 = amin_point[0], amax_point[1]
        atight_box = [p1, p2, p3, p4]
        length, width = amax_point[0] - amin_point[0], amax_point[1] - amin_point[1]
        print(length, width)
        test = 1
        o, r, tm, tc, info = self.env.step([0, 0])

        self.root.after(10, self.environment_step)  # schedule the function to run again after 10 milliseconds

    def setup_ui(self):
        for key, value in self.asset_metainfo.items():
            if value == None:
                continue
            frame = tk.Frame(self.root)
            frame.pack(fill=tk.X, padx=10, pady=5)
            min_val, max_val = self.RANGE_SPECS.get(key, (0, value * 2 if not isinstance(value, tuple) else 2))

            if isinstance(value, (int, float)):
                ttk.Label(frame, text=key).pack(side=tk.LEFT)
                slider_cmd = partial(self.slider_command, key=key)
                slider = ttk.Scale(frame, from_=min_val, to=max_val, value=value, command=slider_cmd, length=700,
                                   orient=tk.HORIZONTAL)
                slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

                # Create an entry next to the slider
                entry_var = tk.StringVar()
                entry_var.set(str(value))
                entry = ttk.Entry(frame, textvariable=entry_var, width=8)
                entry.pack(side=tk.RIGHT, padx=5)
                entry.bind('<Return>', partial(self.entry_command, key=key, entry_var=entry_var))
                self.entries[key] = entry_var
        save_button = ttk.Button(self.root, text="Save", command=self.save_metainfo_to_json)
        save_button.pack(pady=10)
        cancel_button = ttk.Button(self.root, text="Cancel", command=self.cancel_and_exit)
        cancel_button.pack(side=tk.RIGHT, padx=5, pady=10)
    def slider_command(self, v, key, idx=None):
        self.update_value(key, v, idx)
        if idx is not None:
            self.entries[key][idx].set(v)  # Update the entry with the new slider value for tuple
        else:
            self.entries[key].set(v)  # Update the entry with the new slider value for single value

    def entry_command(self, event, key, entry_var, idx=None):
        value = entry_var.get()
        try:
            float_val = float(value)
            self.update_value(key, value, idx)
            # Assuming you keep a reference to your slider widgets, you can also set the slider value here.
        except ValueError:
            print("Invalid value entered.")
    def save_metainfo_to_json(self):
        """Save the modified MetaInfo to a JSON file."""
        if self.save_path is not None:
            with open(self.save_path, 'w') as file:
                json.dump(self.asset_metainfo, file)
        self.run_result = True
        self.root.destroy()
    def cancel_and_exit(self):
        """Close the app without saving."""
        self.root.destroy()
        self.run_result = False
    def run(self):
        self.root.after(10, self.environment_step)
        self.root.mainloop()
        self.env.close()
        return self.run_result

if __name__ == "__main__":
   model_path_input = 'test/vehicle.glb'
   updater = AssetMetaInfoUpdater(model_path_input)
   updater.run()
#
# from metaurban.envs.test_asset_metaurban_env import TestAssetmetaurbanEnv
# from metaurban.examples import expert
#
# # Set the envrionment config
# asset_metainfo = {
#     "TIRE_RADIUS": 0.313,
#     "TIRE_WIDTH": 0.25,
#     "MASS": 1100,
#     "LATERAL_TIRE_TO_CENTER": 0.815,
#     "FRONT_WHEELBASE": 1.05234,
#     "REAR_WHEELBASE": 1.4166,
#     "MODEL_PATH": 'test/vehicle.glb',
#     "MODEL_SCALE": (1, 1, 1),
#     "MODEL_ROTATE": (0, 0.075, 0.),
#     "MODEL_SHIFT": (0, 0, 0),
#     "LENGTH": 4.515,
#     "HEIGHT": 1.139,
#     "WIDTH": 1.852
# }
# env_config={
#     "manual_control": True,
#     "use_render": True,
#     # "controller": "keyboard",  # or joystick
#     "window_size": (1600, 1100),
#     "start_seed": 1000,
#     "test_asset_meta_info": asset_metainfo
#     # "map": "COT",
#     # "environment_num": 1,
# }
#
#
# # ===== Setup the training environment =====
# env = TestAssetmetaurbanEnv(config=env_config)
#
# o, _ = env.reset()
# # env.vehicle.expert_takeover = True
# count = 1
# for i in range(1, 9000000000):
#     o, r, tm, tc, info = env.step(test_asset_config_dict=asset_metainfo,actions={"default_agent": [0,0], "test_agent":[0,0]})