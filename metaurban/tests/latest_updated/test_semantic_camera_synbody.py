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
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
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
        num_scenarios=1,
        accident_prob=0,
        window_size=(960, 960),
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        image_observation=True,
        sensors=dict(
            rgb_camera=(RGBCamera, 256, 256),
            depth_camera=(DepthCamera, 256, 256),
            semantic_camera=(SemanticCamera, 256, 256),
        ),
        agent_observation=ThreeSourceMixObservation,
        interface_panel=[],
    )

    env = SidewalkStaticMetaUrbanEnv(config)
    o, _ = env.reset(seed=0)

    # spawn some pedestrians
    from metaurban.component.agents.pedestrian.pedestrian_type import SimplePedestrian
    from metaurban.component.agents.pedestrian.pedestrian_type import EdogPedestrian
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

    random_humanoid_config = {"spawn_position_heading": [(agent_pos[0], agent_pos[1] + 2), -1.5]}
    edog = env.engine.spawn_object(EdogPedestrian, vehicle_config=random_humanoid_config)
    ped_list.append(ped)

    camera = env.engine.get_sensor("rgb_camera")
    rgb_back = camera.perceive(new_parent_node=env.agent.origin, position=[0, -5, 2], hpr=[0, 0, 0])
    rgb_top = camera.perceive(new_parent_node=env.agent.origin, position=[0, 0, 5], hpr=[0, 270, 0])
    rgb_side = camera.perceive(new_parent_node=env.agent.origin, position=[-5, 0, 2], hpr=[-90, 0, 0])
    max_rgb_value = rgb_back.max()
    rgb = np.concatenate((rgb_back, rgb_top, rgb_side), axis=1)[..., ::-1]
    if max_rgb_value > 1:
        rgb = rgb.astype(np.uint8)
    else:
        rgb = (rgb * 255).astype(np.uint8)

    camera = env.engine.get_sensor("depth_camera")
    depth_back = camera.perceive(
        new_parent_node=env.agent.origin, position=[0, -5, 2], hpr=[0, 0, 0]
    ).reshape(256, 256, -1)[..., -1]
    depth_top = camera.perceive(
        new_parent_node=env.agent.origin, position=[0, 0, 5], hpr=[0, 270, 0]
    ).reshape(256, 256, -1)[..., -1]
    depth_side = camera.perceive(
        new_parent_node=env.agent.origin, position=[-5, 0, 2], hpr=[-90, 0, 0]
    ).reshape(256, 256, -1)[..., -1]
    depth = np.concatenate((depth_back, depth_top, depth_side), axis=1)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    camera = env.engine.get_sensor("semantic_camera")
    semantic_back = camera.perceive(new_parent_node=env.agent.origin, position=[0, -5, 2], hpr=[0, 0, 0])
    semantic_top = camera.perceive(new_parent_node=env.agent.origin, position=[0, 0, 5], hpr=[0, 270, 0])
    semantic_side = camera.perceive(new_parent_node=env.agent.origin, position=[-5, 0, 2], hpr=[-90, 0, 0])
    max_rgb_value = semantic_back.max()
    semantic = np.concatenate((semantic_back, semantic_top, semantic_side), axis=1)[..., ::-1]
    if max_rgb_value > 1:
        semantic = semantic.astype(np.uint8)
    else:
        semantic = (semantic * 255).astype(np.uint8)

    rgbds = np.concatenate((rgb, depth_colored, semantic), axis=0)

    # Display the normalized depth image
    plt.imshow(rgbds)
    plt.title('Paired RGB-Depth-Semantic Image (Back-Top-Side)')
    plt.axis('off')
    plt.show()

    env.close()
