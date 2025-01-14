"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""


import argparse
import logging
import random

import cv2
import numpy as np
from metaurban import SidewalkStaticMetaUrbanEnv, SidewalkDynamicMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.obs.image_obs import ImageObservation
import torch

import yaml
with open('./custom_metaurban_env.yaml', 'r') as f:
    predefined_config = yaml.safe_load(f)

env_type = SidewalkStaticMetaUrbanEnv if predefined_config['Env'] == 'Static' else SidewalkDynamicMetaUrbanEnv


if __name__ == "__main__":
    map_type = predefined_config['Blocks'][0]['Map']
    den_scale = predefined_config['BackgroundAgent'][5]['Total_Number']
    object_density = predefined_config['ObjectDensity']
    config = dict(
        crswalk_density=1,
        object_density=object_density,
        use_render=predefined_config['Rendering'],
        map = map_type, # 5
        manual_control=predefined_config['ManualControl'],
        # traffic_mode = "respawn",
        spawn_human_num=int(predefined_config['BackgroundAgent'][0]['Human_Density'] * den_scale),
        spawn_wheelchairman_num=int(predefined_config['BackgroundAgent'][4]['WheelChair_Density'] * den_scale),
        spawn_edog_num=int(predefined_config['BackgroundAgent'][1]['RobotDog_Density'] * den_scale),
        spawn_erobot_num=int(predefined_config['BackgroundAgent'][2]['Humanoid_Density'] * den_scale),
        spawn_drobot_num=int(predefined_config['BackgroundAgent'][3]['DeliveryRobot_Density'] * den_scale),
        max_actor_num=1,
        drivable_area_extension=55,
        height_scale = 1,
        spawn_deliveryrobot_num=2,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=1000,
        traffic_density=0.6,
        accident_prob=0,
        window_size=(960, 960),
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        
        predefined_config=predefined_config
    )
    
    if predefined_config['ObservationType'] == "all":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 640, 640), depth_camera=(DepthCamera, 640, 640), semantic_camera=(SemanticCamera, 640, 640),),
                agent_observation=ThreeSourceMixObservation,
                interface_panel=[]
            )
        )
    elif predefined_config['ObservationType'] == "rgb":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 640, 640)),
                agent_observation=ImageObservation,
                interface_panel=[]
            )
        )
    elif predefined_config['ObservationType'] == "rgb":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(depth_camera=(DepthCamera, 640, 640)),
                agent_observation=ImageObservation,
                interface_panel=[]
            )
        )
    elif predefined_config['ObservationType'] == "rgb":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(semantic_camera=(SemanticCamera, 640, 640)),
                agent_observation=ImageObservation,
                interface_panel=[]
            )
        )

    env = env_type(config)
    o, _ = env.reset(seed=930)
    
    try:
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0., 0.0])   ### reset; get next -> empty -> have multiple end points

            if (tm or tc):
                env.reset(env.current_seed + 1)
    finally:
        env.close()
