# - Added a test case for the DepthCamera

# Previous version:
# 1. first frame is empty
# 2. fail to use pos and hpr to control the depth camera

# Current version:
# 1. first frame is not empty
# 2. use pos and hpr to control the depth camera successfully

# Changes:
# 1.

from metaurban import SidewalkStaticMetaUrbanEnv
import cv2
import numpy as np
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from matplotlib import pyplot as plt

if __name__ == "__main__":
    map_type = 'X'
    config = dict(
        crswalk_density=1,
        object_density=0.8,
        use_render=True,
        map=map_type,
        manual_control=False,
        drivable_area_extension=55,
        height_scale=1,
        spawn_deliveryrobot_num=2,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=True,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=True,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            lidar=dict(num_lasers=120, distance=50),
            lane_line_detector=dict(num_lasers=0, distance=50),
            side_detector=dict(num_lasers=12, distance=50)
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=10,
        accident_prob=0,
        window_size=(960, 960),
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        image_observation=False,
        interface_panel=[],
    )

    env = SidewalkStaticMetaUrbanEnv(config)
    o, _ = env.reset(seed=0)
    env.engine.toggleDebug()

    # spawn some pedestrians
    from metaurban.component.agents.pedestrian.pedestrian_type import SimplePedestrian
    agent_pos = [env.agents['default_agent'].position[0], env.agents['default_agent'].position[1]]
    pos_list = [
        (agent_pos[0] - 2, agent_pos[1] - 2), (agent_pos[0] - 2, agent_pos[1] + 2),
        (agent_pos[0] + 2, agent_pos[1] + 2), (agent_pos[0] + 2, agent_pos[1] - 2)
    ]
    ped_list = []
    for i, pos in enumerate(pos_list):
        random_humanoid_config = {"spawn_position_heading": [pos, -1.5]}
        ped = env.engine.spawn_object(SimplePedestrian, vehicle_config=random_humanoid_config)
        ped_list.append(ped)

    try:
        for t in range(1, 20000):

            o, r, tm, tc, info = env.step([0., 0.0])

            if (tm or tc):
                for ped in ped_list:
                    ped.before_reset()
                env.reset(env.current_seed + 1)
    finally:
        env.close()
