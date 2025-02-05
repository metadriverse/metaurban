"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
import cv2
import os
import numpy as np
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.constants import HELP_MESSAGE
from metaurban.obs.state_obs import LidarStateObservation
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation
from metaurban.obs.image_obs import ImageObservation, ImageStateObservation
import argparse
import torch

from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from functools import partial
from imitation.policies.serialize import load_policy
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.monitor import Monitor


def make_metadrive_env_fn(env_cfg):
    env = SidewalkStaticMetaUrbanEnv(dict(
        log_level=50,
        **env_cfg,
    ))
    env = Monitor(env)
    return env


import math
import torch.nn as nn
import torch


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 256, 128), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        # self.action_mean.weight.data.mul_(0.1)
        # self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, action_log_std, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action, normal_log_density(action, action_mean, action_log_std, action_std)

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        prob = torch.sigmoid(value)
        return prob


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
    map_type = 'X'
    config = dict(
        crswalk_density=1,
        object_density=0.4,
        use_render=True,
        walk_on_all_regions=False,
        map=map_type,
        manual_control=True,
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
            enable_reverse=True,
            policy_reverse=False,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        window_size=(1200, 900),
        relax_out_of_road_done=True,
        max_lateral_dist=1e10,
        
        camera_dist = 0.8,  # 0.8, 1.71
        camera_height = 1.5,  # 1.5
        camera_pitch = None,
        camera_fov = 66,  # 60, 66
        norm_pixel=False,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="saved_imgs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config.update(
            dict(
                image_observation=True,
                sensors=dict(
                    rgb_camera=(RGBCamera, 1024, 576),
                    depth_camera=(DepthCamera, 1024, 576),
                    semantic_camera=(SemanticCamera, 1024, 576),
                ),
                agent_observation=ThreeSourceMixObservation,
                interface_panel=[]
            )
        )

    env = SidewalkStaticMetaUrbanEnv(config)
    o, _ = env.reset(seed=20)

    action = [0., 0.]
    time_interval = 5
    scenario_t = 0
    try:
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):

            o, r, tm, tc, info = env.step(action)  ### reset; get next -> empty -> have multiple end points
            if scenario_t % time_interval == 0:
                # ===== Prepare input =====
                camera = env.engine.get_sensor("rgb_camera")
                rgb_front = camera.perceive(
                    to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, 2, 2], hpr=[0, 0, 0]
                )
                max_rgb_value = rgb_front.max()
                rgb = rgb_front[..., ::-1]
                if max_rgb_value > 1:
                    rgb = rgb.astype(np.uint8)
                else:
                    rgb = (rgb * 255).astype(np.uint8)

                camera = env.engine.get_sensor("depth_camera")
                depth_front = camera.perceive(
                    to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, 2, 2], hpr=[0, 0, 0]
                ).reshape(576, 1024, -1)[..., -1]
                depth = depth_front
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                depth_img = cv2.bitwise_not(depth_front)
                depth_img = depth_img[..., None]

                camera = env.engine.get_sensor("semantic_camera")
                semantic_front = camera.perceive(
                    to_float=config['norm_pixel'], new_parent_node=env.agent.origin, position=[0, 2, 2], hpr=[0, 0, 0]
                )
                max_rgb_value = semantic_front.max()
                semantic = semantic_front
                if max_rgb_value > 1:
                    semantic = semantic.astype(np.uint8)
                else:
                    semantic = (semantic * 255).astype(np.uint8)
                semantic = semantic[..., ::-1]
                
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_rgb.png"), rgb[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_semantic.png"), semantic[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_depth_colored.png"), depth_colored[..., ::-1])
                cv2.imwrite(os.path.join(args.out_dir, f"seed_{env.current_seed:06d}_time_{scenario_t:06d}_depth_raw.png"), depth_img[..., ::-1])
                
            scenario_t += 1

            if (tm or tc):
                env.reset(env.current_seed + 1)
                action = [0., 0.]
                scenario_t = 0
    finally:
        env.close()
