# Script used to Add new car model and make sure it has proper size and proper tire position.
# Please refer to "CustomizedCar" class in metaurban/component/vehicle/vehicle_type.py for how the car model class is defined
# You don't need to understand how ttk works, just use it!
# Use objverse_change_asset_script.py to call this updater for each newly added car asset.
import tkinter as tk
from tkinter import ttk
from functools import partial
from asset.read_config import configReader
from metaurban.envs.test_asset_metaurban_env import TestAssetmetaurbanEnv
from metaurban.component.static_object.traffic_object import TrafficCone, TrafficWarning
from metaurban.component.vehicle.vehicle_type import CustomizedCar
import json
import os
import fnmatch
class AutoAssetMetaInfoUpdater:
    def __init__(self, model_path, save_path_folder, uid):
        """
        Initialize the AssetMetaInfoUpdater with default configurations and parameters.

        :param model_path: Path to the 3D car model asset.
        :param save_path: (Optional) Path to save updated meta information. If file exists, will load previous data.
        """
        self.setInitValue()
        self.config = configReader()
        self.save_path_folder = save_path_folder
        self.uid = uid
        self.save_path = self.find_uid_file(uid=uid, folder_path=save_path_folder)
        # If we have already validify the parameters and saved it, load it.
        if self.save_path and os.path.exists(os.path.join(self.save_path_folder, self.save_path)):
            with open(os.path.join(self.save_path_folder, self.save_path), 'r') as file:
                loaded_metainfo = json.load(file)
                self.asset_metainfo.update(loaded_metainfo)  # update the asset_metainfo with the values from the file
                self.asset_metainfo["MODEL_SCALE"] = tuple(self.asset_metainfo["MODEL_SCALE"])
                self.asset_metainfo["MODEL_OFFSET"] = tuple(self.asset_metainfo["MODEL_OFFSET"])
                self.asset_metainfo["MODEL_SCALE"] = tuple(self.asset_metainfo["MODEL_SCALE"])
        # Env config
        self.env_config = {
            "manual_control": True,
            "use_render": True,
            "window_size": (1600, 1100),
            "start_seed": 1000,
            "test_asset_meta_info": self.asset_metainfo
        }
        self.env_config["test_asset_meta_info"]["MODEL_PATH"] = model_path
        # Important to use this Env to ensure interactive change of car assets.
        self.env = TestAssetmetaurbanEnv(config=self.env_config)
        o, _ = self.env.reset()
        self.initTK()
        self.env.engine.spawn_object(TrafficWarning, position=[0, 0], heading_theta=0,
                                     random_seed=1)
    def initTK(self):
        self.root = tk.Tk()
        self.root.title("Asset MetaInfo Updater")
        self.detailed_type_var = tk.StringVar()
        self.entries = {}
        self.message_label = ttk.Label(self.root, text="")
        self.message_label.pack(pady=10)
        self.setup_ui()
        self.run_result = None
    def find_uid_file(self, uid, folder_path):
        pattern = f'*-{uid}.json'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            return None
        files = os.listdir(folder_path)
        matching_files = fnmatch.filter(files, pattern)
        return matching_files[0] if matching_files else None

    def slider_command(self, v, key, idx=None):
        """
        Callback function for slider interactions.

        :param v: Value from the slider.
        :param key: The attribute name corresponding to the slider.
        :param idx: (Optional) If the attribute is tuple-like, idx represents which element of the tuple to update.
        """
        self.update_value(key, v, idx)

    def update_value(self, name, value, index=None):
        """
        Update the asset meta information based on user interactions.

        :param name: Name of the attribute to update.
        :param value: New value to set for the attribute.
        :param index: (Optional) If the attribute is tuple-like, index represents which element of the tuple to update.
        """
        constraints = {
            "TIRE_RADIUS": lambda x: x if x > 0 else 0.01,
            "TIRE_WIDTH": lambda x: x if x > 0 else 0.01,
            "MODEL_SCALE": lambda x: x if x != 0 else 1,
        }

        if name.startswith("MODEL_SCALE_"):
            idx = int(name.split("_")[-1])
            current_tuple = list(self.asset_metainfo["MODEL_SCALE"])
            new_val = float(value)
            if "MODEL_SCALE" in constraints:
                new_val = constraints["MODEL_SCALE"](new_val)
            current_tuple[idx] = new_val
            self.asset_metainfo["MODEL_SCALE"] = tuple(current_tuple)
        elif  name.startswith("MODEL_OFFSET_"):
            idx = int(name.split("_")[-1])
            current_tuple = list(self.asset_metainfo["MODEL_OFFSET"])
            new_val = float(value)
            if "MODEL_OFFSET" in constraints:
                new_val = constraints["MODEL_OFFSET"](new_val)
            current_tuple[idx] = new_val
            self.asset_metainfo["MODEL_OFFSET"] = tuple(current_tuple)
        elif  name.startswith("MODEL_HPR_"):
            idx = int(name.split("_")[-1])
            current_tuple = list(self.asset_metainfo["MODEL_HPR"])
            new_val = float(value)
            if "MODEL_HPR" in constraints:
                new_val = constraints["MODEL_HPR"](new_val)
            current_tuple[idx] = new_val
            self.asset_metainfo["MODEL_HPR"] = tuple(current_tuple)
        else:
            if isinstance(self.asset_metainfo[name], tuple):
                new_val = [float(value)] * 3
                self.asset_metainfo[name] = tuple(new_val)
            else:
                new_val = float(value)
                if name in constraints:
                    new_val = constraints[name](new_val)
                self.asset_metainfo[name] = new_val

        # Update the corresponding StringVar in self.entries
        if name in self.entries:
            if isinstance(self.entries[name], dict):  # Check if it's a dictionary
                for idx, entry_var in self.entries[name].items():
                    entry_var.set(value)
            else:
                self.entries[name].set(value)
        elif "MODEL_SCALE" in name:
            if name == "MODEL_SCALE":
                for idx in range(3):
                    self.entries[f"MODEL_SCALE_{idx}"].set(value)
            else:
                if isinstance(self.entries[name], dict):  # Check if it's a dictionary
                    for idx, entry_var in self.entries[name].items():
                        entry_var.set(value)
                else:
                    self.entries[name].set(value)

    def environment_step(self):
        """
        Steps the environment forward using the current configuration and schedules the same function to run again.
        Note: We cannot let it step freely, otherwise we cannot see newly spawned assets
        """
        o, r, tm, tc, info = self.env.step(test_asset_config_dict=self.asset_metainfo,
                                      actions={"default_agent": [0, 0], "test_agent": [0, 0]})
        self.root.after(10, self.environment_step)  # schedule the function to run again after 10 milliseconds
    def getCurrHW(self):
        agents = self.env.engine.agent_manager.spawned_objects
        for id, currcar in agents.items():
            if isinstance(currcar, CustomizedCar):
                self.current_obj = currcar
        amin_point, amax_point = self.current_obj.origin.getTightBounds()
        p1 = amax_point[0], amax_point[1]
        p2 = amax_point[0], amin_point[1]
        p3 = amin_point[0], amin_point[1]
        p4 = amin_point[0], amax_point[1]
        bounding_box = [p1, p2, p3, p4]
        length, width = amax_point[0] - amin_point[0], amax_point[1] - amin_point[1]
        center = [(amax_point[0] + amin_point[0])/2, (amax_point[1] + amin_point[1]) / 2]
        return length, width, bounding_box, center
    def on_detailed_type_selected(self, event):
        detailed_type = self.detailed_type_var.get()
        self.save_path = f"car_{detailed_type}-{self.uid}.json"
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
    def on_update_scale(self):
        computed_scale = (self.dimensions['length'] + self.dimensions['width']) / 2
        length, width, boudingbox, center = self.getCurrHW()
        current_scale = (length + width) / 2
        # self.create_ui_component('scale')
        adjusted_scale =  computed_scale / current_scale
        self.asset_metainfo['MODEL_SCALE'] = (adjusted_scale, adjusted_scale, adjusted_scale)
    def save_width_height_to_yaml(self):
        length, width, bounding_box, center = self.getCurrHW()
        self.config.updateTypeInfo({self.detailed_type_var.get(): {"length":length,
                                                                   "width": width,
                                                                   "bounding_box":bounding_box,
                                                                   "center":center}})
    def setup_ui(self):
        """
        Sets up the graphical user interface elements for the meta info updater.
        """
        ttk.Label(self.root, text="Select Detailed Type:").pack(pady=10)
        self.detailed_type_dropdown = ttk.Combobox(self.root, textvariable=self.detailed_type_var)
        self.detailed_type_dropdown['values'] = list(self.config.loadCarType())
        self.detailed_type_dropdown.bind('<<ComboboxSelected>>', self.on_detailed_type_selected)
        self.detailed_type_dropdown.pack(pady=10)

        # About Color
        self.color_var = tk.StringVar()
        self.color_list = self.config.loadColorList()

        ttk.Label(self.root, text="Select Color:").pack(pady=10)
        color_dropdown = ttk.Combobox(self.root, textvariable=self.color_var, values=self.color_list)
        color_dropdown.pack(pady=10)
        for key, value in self.asset_metainfo.items():
            if key == "general":
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

            elif isinstance(value, tuple):

                if key == "MODEL_SCALE":
                    # Slider to control all 3 elements together
                    sub_frame_all = tk.Frame(self.root)
                    sub_frame_all.pack(fill=tk.X, padx=10, pady=5)
                    # For controlling all 3 elements together
                    ttk.Label(sub_frame_all, text=key).pack(side=tk.LEFT)
                    slider_cmd = partial(self.slider_command, key=key)
                    slider = ttk.Scale(sub_frame_all, from_=min_val, to=max_val, value=value[0], command=slider_cmd,
                                       length=700, orient=tk.HORIZONTAL)
                    slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                    entry_var = tk.StringVar()
                    entry_var.set(str(value[0]))  # you might consider showing all 3 values in some formatted way
                    entry = ttk.Entry(sub_frame_all, textvariable=entry_var, width=8)
                    entry.pack(side=tk.RIGHT, padx=5)
                    entry.bind('<Return>', partial(self.entry_command, key=key, entry_var=entry_var))
                    self.entries[key] = entry_var  # This is a flattened structure

                    # Individual sliders for each tuple element
                    for sub_idx, val in enumerate(value):
                        sub_key = f"{key}_{sub_idx}"
                        sub_frame = tk.Frame(self.root)
                        sub_frame.pack(fill=tk.X, padx=10, pady=5)
                        ttk.Label(sub_frame, text=sub_key).pack(side=tk.LEFT)
                        slider_cmd = partial(self.slider_command, key=sub_key)  # Change here
                        slider = ttk.Scale(sub_frame, from_=min_val, to=max_val, value=val, command=slider_cmd,
                                           length=700, orient=tk.HORIZONTAL)
                        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                        entry_var = tk.StringVar()
                        entry_var.set(str(val))
                        entry = ttk.Entry(sub_frame, textvariable=entry_var, width=8)
                        entry.pack(side=tk.RIGHT, padx=5)
                        entry.bind('<Return>',
                                   partial(self.entry_command, key=sub_key, entry_var=entry_var))  # Change here
                        self.entries[sub_key] = entry_var  # This will create keys like MODEL_SCALE_0, MODEL_SCALE_1,
                else:
                    for sub_idx, val in enumerate(value):
                        sub_key = f"{key}_{sub_idx}"
                        sub_frame = tk.Frame(self.root)
                        sub_frame.pack(fill=tk.X, padx=10, pady=5)
                        ttk.Label(sub_frame, text=sub_key).pack(side=tk.LEFT)
                        slider_cmd = partial(self.slider_command, key=sub_key)  # Change here
                        slider = ttk.Scale(sub_frame, from_=min_val, to=max_val, value=val, command=slider_cmd,
                                           length=700, orient=tk.HORIZONTAL)
                        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                        entry_var = tk.StringVar()
                        entry_var.set(str(val))
                        entry = ttk.Entry(sub_frame, textvariable=entry_var, width=8)
                        entry.pack(side=tk.RIGHT, padx=5)
                        entry.bind('<Return>',
                                   partial(self.entry_command, key=sub_key, entry_var=entry_var))  # Change here
                        self.entries[sub_key] = entry_var  # This will create keys like MODEL_SCALE_0, MODEL_SCALE_1,
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
        # print(bounding_box)
        print(center)
        print(self.asset_metainfo["MODEL_OFFSET"])
        self.asset_metainfo["MODEL_OFFSET"] = (self.asset_metainfo["MODEL_OFFSET"][0] + center[0],
                                                    self.asset_metainfo["MODEL_OFFSET"][1] + center[1],
                                                    self.asset_metainfo["MODEL_OFFSET"][2])
    def slider_command(self, v, key, idx=None):
        """
        Callback for when a slider is changed. Updates both the value in the backend and the associated entry.

        :param v: The new value from the slider.
        :param key: The attribute name the slider is linked to.
        :param idx: (Optional) Index of the tuple if the attribute is tuple-like.
        """
        self.update_value(key, v, idx)
        if idx is not None:
            self.entries[key][idx].set(v)  # Update the entry with the new slider value for tuple
        else:
            self.entries[key].set(v)  # Update the entry with the new slider value for single value

    def entry_command(self, event, key, entry_var, idx=None):
        """
        Callback function for when an entry field is edited and submitted (e.g., by pressing 'Enter').

        :param event: The event object (not used but required by Tkinter).
        :param key: The attribute name the entry is linked to.
        :param entry_var: The StringVar instance tied to the entry.
        :param idx: (Optional) Index of the tuple if the attribute is tuple-like.
        """
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
            "general_type": "vehicle",
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
    def setInitValue(self):
        # Default car parameters
        self.asset_metainfo = {
            "TIRE_RADIUS": 0.313,
            "TIRE_WIDTH": 0.25,
            "MASS": 1100,
            "LATERAL_TIRE_TO_CENTER": 0.815,
            "FRONT_WHEELBASE": 1.05234,
            "REAR_WHEELBASE": 1.4166,
            "MODEL_PATH": 'test/vehicle.glb',
            "MODEL_SCALE": (1, 1, 1),
            "MODEL_OFFSET": (0, 0, 0),
            "MODEL_HPR": (0, 0, 0),
            "LENGTH": 4.515,
            "HEIGHT": 1.139,
            "WIDTH": 1.852
        }
        # Valid parameter ranges
        self.RANGE_SPECS = {
            "TIRE_RADIUS": (0.01, 1),
            "TIRE_WIDTH": (0.01, 1),
            "LATERAL_TIRE_TO_CENTER": (-2, 2),
            "FRONT_WHEELBASE": (-2, 2),
            "REAR_WHEELBASE": (-2, 2),
            "MODEL_SCALE": (-2, 2),
            "MODEL_OFFSET": (-4, 4),
            "MODEL_HPR": (-180, 180)
        }

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