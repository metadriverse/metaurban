"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.constants import HELP_MESSAGE
import cv2
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
    env = SidewalkStaticMetaUrbanEnv(
        dict(
            log_level=50,
            **env_cfg,
        )
    )
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
    map_type = 'C'
    config = dict(
        crswalk_density=1,
        object_density=0.6,
        training=False,
        use_render=True,
        map = map_type,
        manual_control=False,
        drivable_area_extension=75,
        height_scale = 1,
        spawn_deliveryrobot_num=2,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=1000,
        on_continuous_line_done=False,
        out_of_route_done=False,
        vehicle_config=dict(
            show_lidar=True,
            show_navi_mark=True,
            show_line_to_navi_mark=True,
            show_dest_mark=True,
            # lidar=dict(
            #     num_lasers=0, distance=10, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
            # ),
            # side_detector=dict(num_lasers=0, distance=10, gaussian_noise=0.0, dropout_prob=0.0),
            # lane_line_detector=dict(num_lasers=0, distance=5, gaussian_noise=0.0, dropout_prob=0.0),
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=60,
        accident_prob=0,
        window_size=(960, 960),
        relax_out_of_road_done=True,
        max_lateral_dist=10.0,
        
        # physics_world_step_size=1e-1,
        # decision_repeat=1,
        # image_on_cuda=True,
        # camera_smooth_buffer_size=1,
        # shadow_range=1,
        # show_fps=False,
        # show_logo=False,
        # show_mouse=True,
        # show_skybox=False,
        # show_terrain=False,
        # show_interface=False,
        
        # force_render_fps=90,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", 'all', 'rgb'])
    parser.add_argument("--policy", type=str, default="./pretrained_policy_576k")
    args = parser.parse_args()

    if args.observation == "all":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 1920, 1080), depth_camera=(DepthCamera, 640, 640), semantic_camera=(SemanticCamera, 640, 640),),
                agent_observation=ThreeSourceMixObservation,
                interface_panel=[]
            )
        )
    if args.observation == "rgb":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 1920, 1080)),
                agent_observation=ImageStateObservation,
                interface_panel=[]
            )
        )
    
    env = SidewalkStaticMetaUrbanEnv(config)
    o, _ = env.reset(seed=0)
    n_envs = 1
    
    
    algo_config=dict(
        learning_rate=5e-5,
        n_steps=200,
        batch_size=256,
        n_epochs=10,
        vf_coef=1.0,
        max_grad_norm=10.0,
        verbose=1,
        seed=0,
        ent_coef=0.0,
        tensorboard_log="./metaurban_ppo-single_scenario_per_process_1e8-tb_logs/",
    )
    env_fn = make_metadrive_env_fn
    expert = PPO(
        "MlpPolicy",
        env,
        **algo_config,
    )
    load_path_or_dict = args.policy.split('.zip')[0]
    from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
    _, params, _ = load_from_zip_file(load_path_or_dict, device='cpu', load_data=False)
    expert.set_parameters(params, exact_match=True, device='cpu')

    action = [0., 0.]
    route_completed_list = []
    sr_list = []
    cost_list = []
    cost = 0.0
    not_comply_time = 0.0
    sns_list = []
    spl_list = []
    dis = 0.0

    last_pos = None
    try:
        print(HELP_MESSAGE)
        for i in range(1, 1000000000):
                
            o, r, tm, tc, info = env.step(action)   ### reset; get next -> empty -> have multiple end points
            #action = expert.select_action(torch.from_numpy(o).reshape(1, 271))[0].detach().numpy()
            
            action = expert.predict(torch.from_numpy(o).reshape(1, 271))[0] #.detach().numpy()
            action = np.clip(action, a_min=-1, a_max=1.)
            action = action[0].tolist()
            
            if info['crash_vehicle'] > 0:
                cost += 1.0
                not_comply_time += 1 
            if info['crash_object'] > 0:
                cost += 1.0
            if info['crash_human'] > 0:
                cost += 1.0
                
            if info['crash_vehicle'] or info['crash_object'] or info['crash_human']:
                not_comply_time += 1.0
            if last_pos is not None:
                dis += np.linalg.norm(np.array(last_pos).reshape(2,) - np.array(env.agents['default_agent'].position)[..., :2].reshape(2,))
            last_pos = np.array(env.agents['default_agent'].position)[..., :2].reshape(2,)

            if (tm or tc):
                sns_list.append(1 - not_comply_time / info['episode_length'])
                sr_list.append(int(info['is_success']))
                if int(info['is_success']):
                    spl_list.append(min(env.agents['default_agent'].navigation.reference_trajectory.length / dis, 1.))
                else:
                    spl_list.append(0)
                route_completed_list.append(info['route_completion'])
                cost_list.append(cost)
                if len(cost_list) % 10 == 0:
                    print('Success Rate:', sum(sr_list) / len(sr_list), 'SPL:', sum(spl_list) / len(spl_list),
                          'Route Completion:', sum(route_completed_list) / len(route_completed_list), 'Cost:', sum(cost_list) / len(cost_list), 'SNS:', sum(sns_list) / len(sns_list))
                env.reset(env.current_seed + 1)
                action = [0., 0.]
                cost = 0.0
                not_comply_time = 0.0
                dis = 0.0
    finally:
        env.close()
