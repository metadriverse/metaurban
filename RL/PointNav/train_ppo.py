import gymnasium as gym
import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo.ppo import PPO
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.constants import HELP_MESSAGE

from metaurban.obs.state_obs import LidarStateObservation
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument('--unique_id', type=int, default=0)
args = parser.parse_args()

set_random_seed(args.unique_id)
import os
os.makedirs('./RL_logs', exist_ok=True)
os.makedirs('./RL_logs/PPO', exist_ok=True)
exptid = f"{args.unique_id:04d}"

config = dict(
    env=dict(
        use_render=False,
        # This policy setting simplifies the task
        # NOTE: do not use discrete action
        # discrete_action=True,
        # discrete_throttle_dim=3,
        # discrete_steering_dim=3, 
        map="X",
        training=True,
        object_density=0.6,
        crswalk_density=1,
        spawn_human_num=10,
        spawn_robotdog_num=10,
        spawn_deliveryrobot_num=10,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=1000,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=True,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            use_saver=False, overtake_stat=False
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=1000,
        traffic_density=0,
        accident_prob=0,
        crash_vehicle_done=False,
        crash_object_done=False,
        relax_out_of_road_done=True,
        drivable_area_extension=75,
        
        # ===== Reward Scheme =====
        # See: https://github.com/metaurbanrse/metaurban/issues/283
        success_reward=8.0,
        out_of_road_penalty=3.0,
        on_lane_line_penalty=1.,
        crash_vehicle_penalty=2.,
        crash_object_penalty=2.0,
        crash_human_penalty=2.0,
        crash_building_penalty=2.0,
        driving_reward=2.0,
        steering_range_penalty=2.0,
        heading_penalty=0.0,
        lateral_penalty=2.0,
        max_lateral_dist=5.,
        speed_reward=0.5,
        no_negative_reward=True,

        # ===== Cost Scheme =====
        crash_vehicle_cost=2.0,
        crash_object_cost=2.0,
        out_of_road_cost=2.0,
        crash_human_cost=2.0,
        agent_observation=LidarStateObservation
    ),
    algo=dict(
        learning_rate=5e-4,
        n_steps=200,
        batch_size=256,
        n_epochs=10,
        vf_coef=1.0,
        max_grad_norm=10.0,
        verbose=1,
        seed=0,
        ent_coef=0.0,
        tensorboard_log=f"./RL_logs/PPO/metaurban_point_nav_{exptid}-tb_logs/",
    ),
    n_envs=20,
)
env_cfg = config["env"]


def make_metaurban_env_fn(env_cfg, seed):
    env = SidewalkStaticMetaUrbanEnv(
        dict(
            start_seed=seed,
            log_level=50,
            **env_cfg,
        )
    )
    env = Monitor(env)
    return env


def train():
    env_fn = make_metaurban_env_fn

    env_cfg = config["env"]

    env = SubprocVecEnv(
        [partial(env_fn, env_cfg, seed) for seed in range(config["n_envs"])]
    )
    # env._seeds = [i * 100 for i in range(config["n_envs"])]
    import copy
    eval_cfg = copy.deepcopy(env_cfg)
    eval_cfg['training'] = False
    eval_env = SubprocVecEnv(
        [
            partial(env_fn, eval_cfg, seed)
            for seed in range(950, 999)
        ]
    )

    # env = SubprocVecEnv([partial(env_fn, env_cfg, 0)])
    # eval_env = SubprocVecEnv([partial(env_fn, env_cfg, 1)])

    model = PPO(
        "MlpPolicy",
        env,
        **config["algo"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e4),
        save_path=f"./RL_logs/PPO/metaurban_point_nav_{exptid}_ckpt_logs/",
        name_prefix=exptid,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./RL_logs/PPO/metaurban_point_nav_{exptid}_best_model_logs/",
        log_path=f"./RL_logs/PPO/metaurban_point_nav_{exptid}_eval_logs/",
        eval_freq=int(1e3),
        deterministic=True,
        render=False,
    )

    # wandb_callback = WandbCallback()

    # callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model.learn(
        total_timesteps=int(1e8),
        callback=callbacks,
        # log_interval=4,
        # verbose=1,
    )
    env.close()


if __name__ == "__main__":
    train()
    