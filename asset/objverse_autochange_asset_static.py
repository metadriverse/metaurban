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
import fnmatch
import yaml
from asset.read_config import configReader
from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.component.static_object.test_new_object import TestObject, TestGLTFObject
from metaurban.component.static_object.traffic_object import TrafficCone, TrafficWarning
class AutoStaticAssetMetaInfoUpdater:
    def __init__(self, file_name,  save_path_folder=None, uid = None, folder_name=None, isGLTF = False):
        self.setInitValue(file_name, folder_name)
        self.save_path_folder = save_path_folder
        self.uid = uid
        self.save_path = self.find_uid_file(uid = uid, folder_path = save_path_folder)
        if self.save_path:
            print(self.save_path)
            with open(os.path.join(self.save_path_folder, self.save_path), 'r') as file:
                loaded_metainfo = json.load(file)
                self.asset_metainfo.update(loaded_metainfo)  # update the asset_metainfo with the values from the file

        self.isGLTF = isGLTF
        self.entries = {}
        self.config = configReader()
        self.types = self.config.loadType()
        self.env = metaurbanEnv(config=self.env_config)
        o, _ = self.env.reset()
        self.initTK()
        self.setup_ui()
        self.run_result = None
        self.current_obj = None
        self.env.engine.spawn_object(TrafficWarning, position=[0, 0], heading_theta=0,
                                     random_seed=1)

    def find_uid_file(self, uid, folder_path):
        pattern = f'*-{uid}.json'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            return None
        files = os.listdir(folder_path)
        matching_files = fnmatch.filter(files, pattern)
        return matching_files[0] if matching_files else None
    def initTK(self):
        self.root = tk.Tk()
        self.root.title("Asset MetaInfo Updater")
        self.general_type_var = tk.StringVar()
        self.detailed_type_var = tk.StringVar()
        self.message_label = ttk.Label(self.root, text="")
        self.message_label.pack(pady=10)
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
            elif isinstance(self.entries[name], float) or isinstance(self.entries[name], str):
                self.entries[name] = value
            else:
                self.entries[name].set(value)

    def environment_step(self):
        """Steps through the environment with the updated asset_metainfo configuration and schedules itself to run again."""
        if self.current_obj is not None:
            self.env.engine.clear_objects([self.current_obj.id], force_destroy=True)
        if self.isGLTF:
            self.current_obj = self.env.engine.spawn_object(TestGLTFObject, position=[0, 0], heading_theta=0,
                                                            random_seed=1, force_spawn=True,
                                                            asset_metainfo=self.asset_metainfo)
        else:
            self.current_obj = self.env.engine.spawn_object(TestObject, position=[0, 0], heading_theta=0, random_seed=1, force_spawn=True,
                                    asset_metainfo=self.asset_metainfo)
        o, r, tm, tc, info = self.env.step([0, 0])

        self.root.after(10, self.environment_step)  # schedule the function to run again after 10 milliseconds
    def getCurrHW(self):
        if self.current_obj is None:
            if self.isGLTF:
                self.current_obj = self.env.engine.spawn_object(TestGLTFObject, position=[0, 0], heading_theta=0,
                                                                random_seed=1, force_spawn=True,
                                                                asset_metainfo=self.asset_metainfo)
            else:
                self.current_obj = self.env.engine.spawn_object(TestObject, position=[0, 0], heading_theta=0,
                                                                random_seed=1, force_spawn=True,
                                                                asset_metainfo=self.asset_metainfo)
        amin_point, amax_point = self.current_obj.origin.getTightBounds()
        p1 = amax_point[0], amax_point[1]
        p2 = amax_point[0], amin_point[1]
        p3 = amin_point[0], amin_point[1]
        p4 = amin_point[0], amax_point[1]
        bounding_box = [p1, p2, p3, p4]
        length, width = amax_point[0] - amin_point[0], amax_point[1] - amin_point[1]
        center = [(amax_point[0] + amin_point[0])/2, (amax_point[1] + amin_point[1]) / 2]
        return length, width, bounding_box, center
    def on_general_type_selected(self, event):
        # Populate detailed types based on selected general type
        general_type = self.general_type_var.get()
        detailed_types = self.types.get(general_type, [])
        self.detailed_type_dropdown['values'] = detailed_types
        if detailed_types:
            self.detailed_type_var.set(detailed_types[0])  # Set default value
        else:
            print("Warning: No Type for {}".format(general_type))
    def on_update_scale(self):
        computed_scale = (self.dimensions['length'] + self.dimensions['width']) / 2
        length, width, boudingbox, center = self.getCurrHW()
        current_scale = (length + width) / 2
        # self.create_ui_component('scale')
        self.asset_metainfo['scale'] = computed_scale / current_scale
        self.entries['scale'] = self.asset_metainfo['scale']
    def on_detailed_type_selected(self, event):
        detailed_type = self.detailed_type_var.get()
        self.save_path = f"{self.general_type_var.get()}_{detailed_type}-{self.uid}.json"
        # Check if dimensions exist in the YAML data
        self.dimensions = self.config.loadTypeInfo().get(detailed_type)
        if self.dimensions:
            # Set the scale or dimensions in the UI accordingly
            # Chenda:TODO: Add logic to set the scale in UI based on YAML data
            self.message_label.config(text=f"Dimensions found for {detailed_type}: {self.dimensions['length']} x {self.dimensions['width']}")
            self.on_update_scale()
            save_button = ttk.Button(self.root, text="Update Scale", command=self.on_update_scale)
            save_button.pack(pady=10)
            save_button = ttk.Button(self.root, text="Save to YAML", command=self.save_width_height_to_yaml)
            save_button.pack(pady=10)

        else:
            # Allow the user to manually adjust the scale
            # TODO: Add logic for manual scale adjustment
            self.message_label.config(text=f"Dimensions not found for {detailed_type}")
            # self.create_ui_component('scale')
            save_button = ttk.Button(self.root, text="Save to YAML", command=self.save_width_height_to_yaml)
            save_button.pack(pady=10)
    def save_width_height_to_yaml(self):
        length, width, bounding_box, center = self.getCurrHW()
        self.config.updateTypeInfo({self.detailed_type_var.get(): {"length":length,
                                                                   "width": width,
                                                                   "bounding_box":bounding_box,
                                                                   "center":center}})


    def create_ui_component(self, key):
        """Creates a slider and entry widget for the given key."""
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.X, padx=10, pady=5)

        min_val, max_val = self.RANGE_SPECS.get(key, (0, 20))  # Adjust range if necessary

        ttk.Label(frame, text=key).pack(side=tk.LEFT)
        slider_cmd = partial(self.slider_command, key=key)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, value=self.asset_metainfo.get(key, 1),
                           command=slider_cmd, length=700, orient=tk.HORIZONTAL)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        entry_var = tk.StringVar()
        entry_var.set(str(self.asset_metainfo.get(key, 1)))
        entry = ttk.Entry(frame, textvariable=entry_var, width=8)
        entry.pack(side=tk.RIGHT, padx=5)
        entry.bind('<Return>', partial(self.entry_command, key=key, entry_var=entry_var))
        self.entries[key] = entry_var
    def setup_ui(self):
        # Dropdown for general type
        ttk.Label(self.root, text="Select General Type:").pack(pady=10)
        general_type_dropdown = ttk.Combobox(self.root, textvariable=self.general_type_var)
        general_type_dropdown['values'] = list(self.types.keys())
        general_type_dropdown.bind('<<ComboboxSelected>>', self.on_general_type_selected)
        general_type_dropdown.pack(pady=10)

        # Dropdown for detailed type
        ttk.Label(self.root, text="Select Detailed Type:").pack(pady=10)
        self.detailed_type_dropdown = ttk.Combobox(self.root, textvariable=self.detailed_type_var)
        self.detailed_type_dropdown.bind('<<ComboboxSelected>>', self.on_detailed_type_selected)
        self.detailed_type_dropdown.pack(pady=10)

        # About Color
        self.color_var = tk.StringVar()
        self.color_list = self.config.loadColorList()

        ttk.Label(self.root, text="Select Color:").pack(pady=10)
        color_dropdown = ttk.Combobox(self.root, textvariable=self.color_var, values=self.color_list)
        color_dropdown.pack(pady=10)


        for key, value in self.asset_metainfo.items():
            if value is not None and isinstance(value, (int, float)):  # Exclude scale since it's handled separately
                self.create_ui_component(key)
        make_center_button = ttk.Button(self.root, text="Make Center", command=self.make_center)
        make_center_button.pack(pady=10)
        save_button = ttk.Button(self.root, text="Save", command=self.save_metainfo_to_json)
        save_button.pack(pady=10)
        next_button = ttk.Button(self.root, text="Next", command=self.no_change_and_exit)
        next_button.pack(pady=10)
        cancel_button = ttk.Button(self.root, text="Cancel", command=self.cancel_and_exit)
        cancel_button.pack(side=tk.RIGHT, padx=5, pady=10)
    def make_center(self):
        length, width, bounding_box, center = self.getCurrHW()
        self.asset_metainfo['pos0'] -= center[0]
        self.asset_metainfo['pos1'] -= center[1]

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
        self.save_width_height_to_yaml()
        length, width, bounding_box, center = self.getCurrHW()
        general_info_dict = {
            "length": length,
            "width": width,
            "bounding_box": bounding_box,
            "center": center,
            "color": self.color_var.get(),
            "general_type": self.general_type_var.get(),
            "detail_type":self.detailed_type_var.get()
        }
        self.asset_metainfo["general"] = general_info_dict
        if self.save_path is not None:
            with open(os.path.join(self.save_path_folder, self.save_path), 'w') as file:
                json.dump(self.asset_metainfo, file)
        self.run_result = True
        self.root.destroy()
    def no_change_and_exit(self):
        self.root.destroy()
        self.run_result = True
    def cancel_and_exit(self):
        """Close the app without saving."""
        self.root.destroy()
        self.run_result = False
    def run(self):
        self.root.after(10, self.environment_step)
        self.root.mainloop()
        self.env.close()
        return self.run_result
    def setInitValue(self, file_name, folder_name):
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
            "hshift": (-180, 180),
            "pos0": (-10, 10),
            "pos1": (-10, 10),
            "pos2": (-10, 10),
            "scale": (0.01, 20),
        }

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
if __name__ == "__main__":
   model_path_input = 'test/vehicle.glb'
   updater = AutoAssetMetaInfoUpdater(model_path_input)
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