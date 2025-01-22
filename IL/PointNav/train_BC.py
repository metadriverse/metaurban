import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
rng = np.random.default_rng()
import gymnasium as gym

import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo.ppo import PPO
from metaurban import SidewalkStaticMetaDriveEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.policies.serialize import load_policy
from metaurban.obs.state_obs import LidarStateObservation


SEED = 0

import os, pickle
import numpy as np
import tqdm
from imitation.data.types import TrajectoryWithRew
expert_data_path = ''
rollouts_files = os.listdir(expert_data_path)
rollouts = []

import functools
import logging
import pathlib
from typing import Any, Mapping, Optional, Type

import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_adversarial_ex
from imitation.scripts.ingredients import demonstrations, environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward, rl

def save(trainer, save_path: pathlib.Path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    save_path.mkdir(parents=True, exist_ok=True)
    # th.save(trainer.reward_train, save_path / "reward_train.pt")
    # th.save(trainer.reward_test, save_path / "reward_test.pt")
    serialize.save_stable_model(
        save_path / "policy",
        trainer.policy,
    )
    
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
    ),
    n_envs=1,
)
config['env'].update(
    dict(
        image_observation=True,
        sensors=dict(depth_camera=(DepthCamera, 400, 300), rgb_camera=(RGBCamera, 400, 300)),
        interface_panel=["depth_camera", 'rgb_camera', "dashboard"]
    )
)
config['env'].update(
    dict(
        agent_observation=LidarStateObservation
    )
)
from functools import partial

def make_metaurban_env_fn(env_cfg, seed):
    env = SidewalkStaticMetaDriveEnv(
        dict(
            start_seed=seed,
            log_level=50,
            **env_cfg,
        )
    )
    env = Monitor(env)
    return env

checkpoint_interval = 5
def callback(round_num: int, /) -> None:
    if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
        save(bc_trainer, log_dir / "checkpoints" / f"{round_num:05d}")

total = 50000
if __name__ == '__main__':
    env_fn = make_metaurban_env_fn
    env_cfg = config["env"]
    env = SubprocVecEnv(
        [partial(env_fn, env_cfg, seed) for seed in range(config["n_envs"])]
    )
    
    num = 0
    for rollout_dir in tqdm.tqdm(rollouts_files):
        obs = []
        acts = []
        infos = []
        rews = []
        terminal = True
        if num >= total:
            break
        
        t = 0
        obs_t = pickle.load(open(os.path.join(expert_data_path, rollout_dir, f'{t:05d}.pkl'), 'rb'))['obs']['state']
        obs.append(obs_t)
        for t in range(1, len(os.listdir(os.path.join(expert_data_path, rollout_dir)))):
            step_info = pickle.load(open(os.path.join(expert_data_path, rollout_dir, f'{t:05d}.pkl'), 'rb'))
            obs_t = step_info['obs']['state']
            rew_t = step_info['reward']
            acts_t = step_info['info']['action']
            obs.append(obs_t)
            acts.append(acts_t)
            rews.append(rew_t)
            num += 1
            if num >= total:
                break
        try:
            traj_with_rew = TrajectoryWithRew(
                obs=np.array(obs),
                acts=np.array(acts),
                rews=np.array(rews).astype(float),
                terminal=terminal,
                infos=None
            )
            
            rollouts.append(traj_with_rew)
        except:
            continue
        
    transitions = rollout.flatten_trajectories(rollouts)
        
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    # evaluate the learner before training
    env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        bc_trainer.policy, env, 20, return_episode_rewards=True,
    )
    
    print("mean reward before training:", 
          np.mean(learner_rewards_before_training), 
          "+/-",
          np.std(learner_rewards_before_training),)
    
    
    # custom_logger, log_dir = logging_ingredient.setup_logging()
    log_dir = pathlib.Path(f'./BC_{total:07d}')

    # train the learner and evaluate again
    bc_trainer.train(n_epochs=20)  # Train for 800_000 steps to match expert.
    
    env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        bc_trainer.policy, env, 20, return_episode_rewards=True,
    )

    print("mean reward after training:", 
          np.mean(learner_rewards_after_training), 
          "+/-",
          np.std(learner_rewards_after_training),)
    
    save(bc_trainer, log_dir / "checkpoints" / "final")
    