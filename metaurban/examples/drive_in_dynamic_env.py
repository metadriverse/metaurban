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
from metaurban import SidewalkDynamicMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
import torch
"""
Block Type	    ID
Straight	    S  
Circular	    C   #
InRamp	        r   #
OutRamp	        R   #
Roundabout	    O	#
Intersection	X
Merge	        y	
Split	        Y   
Tollgate	    $	
Parking lot	    P.x
TInterection	T	
Fork	        WIP
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", 'all'])
    parser.add_argument("--density_obj", type=float, default=0.3)
    parser.add_argument("--density_ped", type=float, default=1.0)
    args = parser.parse_args()

    map_type = 'X'
    den_scale = args.density_ped
    config = dict(
        crswalk_density=1,
        object_density=args.density_obj,
        walk_on_all_regions=False,
        use_render=True,
        map=map_type,
        manual_control=True,
        drivable_area_extension=55,
        height_scale=1,
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
            enable_reverse=True,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        
        spawn_human_num=int(20 * den_scale),
        spawn_wheelchairman_num=int(1 * den_scale),
        spawn_edog_num=int(2 * den_scale),
        spawn_erobot_num=int(1 * den_scale),
        spawn_drobot_num=int(1 * den_scale),
        max_actor_num=20,
        
        window_size=(1200, 900),
    )

    if args.observation == "all":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(
                    rgb_camera=(RGBCamera, 1920, 1080),
                    depth_camera=(DepthCamera, 640, 640),
                    semantic_camera=(SemanticCamera, 640, 640),
                ),
                agent_observation=ThreeSourceMixObservation,
                interface_panel=[]
            )
        )

    env = SidewalkDynamicMetaUrbanEnv(config)
    o, _ = env.reset(seed=30)

    try:
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0., 0.0])  ### reset; get next -> empty -> have multiple end points

            if (tm or tc):
                env.reset(((env.current_seed + 1) % config['num_scenarios']) + env.engine.global_config['start_seed'])
    finally:
        env.close()
