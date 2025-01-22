import json
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from typing import Dict
import math

import gym
import numpy as np
from metaurban.obs.observation_base import BaseObservation as ObservationBase
from metaurban.utils import clip
from ray.rllib import SampleBatch

from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.offline import InputReader
from ray.rllib.policy import Policy
import logging


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
            self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(
            self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            **kwargs
    ):
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["cost"] = np.nan
        if "custom_metrics" not in result:
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]


class EGPOCallbacks(DrivingCallbacks):
    def on_episode_start(
            self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["takeover"] = 0
        episode.user_data["raw_episode_reward"] = 0
        episode.user_data["episode_crash_rate"] = 0
        episode.user_data["episode_out_of_road_rate"] = 0
        episode.user_data["high_speed_rate"] = 0
        episode.user_data["total_takeover_cost"] = 0
        episode.user_data["total_native_cost"] = 0
        episode.user_data["cost"] = 0
        episode.user_data["episode_crash_vehicle"] = 0
        episode.user_data["episode_crash_object"] = 0

    def on_episode_step(
            self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["takeover"] += 1 if info["takeover"] else 0
            episode.user_data["raw_episode_reward"] += info["step_reward"]
            episode.user_data["episode_crash_rate"] += 1 if info["crash"] else 0
            episode.user_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
            # episode.user_data["high_speed_rate"] += 1 if info["high_speed"] else 0
            episode.user_data["total_takeover_cost"] += info["takeover_cost"]
            episode.user_data["total_native_cost"] += info["native_cost"]
            episode.user_data["cost"] += info["cost"] if "cost" in info else info["native_cost"]

            episode.user_data["episode_crash_vehicle"] += 1 if info["crash_vehicle"] else 0
            episode.user_data["episode_crash_object"] += 1 if info["crash_object"] else 0

    def on_episode_end(
            self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
            **kwargs) -> None:
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["takeover_rate"] = float(episode.user_data["takeover"] / episode.length)
        episode.custom_metrics["takeover_count"] = float(episode.user_data["takeover"])
        episode.custom_metrics["raw_episode_reward"] = float(episode.user_data["raw_episode_reward"])
        episode.custom_metrics["episode_crash_num"] = float(episode.user_data["episode_crash_rate"])
        episode.custom_metrics["episode_out_of_road_num"] = float(episode.user_data["episode_out_of_road_rate"])
        episode.custom_metrics["high_speed_rate"] = float(episode.user_data["high_speed_rate"] / episode.length)

        episode.custom_metrics["total_takeover_cost"] = float(episode.user_data["total_takeover_cost"])
        episode.custom_metrics["total_native_cost"] = float(episode.user_data["total_native_cost"])

        episode.custom_metrics["cost"] = float(episode.user_data["cost"])
        episode.custom_metrics["overtake_num"] = int(episode.last_info_for()["overtake_vehicle_num"])

        episode.custom_metrics["episode_crash_vehicle_num"] = float(episode.user_data["episode_crash_vehicle"])
        episode.custom_metrics["episode_crash_object_num"] = float(episode.user_data["episode_crash_object"])

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["cost"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["takeover"] = np.nan
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
            result["native_cost"] = result["custom_metrics"]["total_native_cost_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]
        if "takeover_count_mean" in result["custom_metrics"]:
            result["takeover"] = result['custom_metrics']["takeover_count_mean"]


# turn on overtake stata only in evaluation
evaluation_config = dict(env_config=dict(
    vehicle_config=dict(use_saver=False, overtake_stat=False),
    safe_rl_env=True,
    start_seed=500,
    environment_num=50,
    horizon=1000,
))


class ILCallBack(EGPOCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["cost"] = np.nan
        result["length"] = np.nan
        result["takeover"] = np.nan
        if "evaluation" in result:
            eval = result["evaluation"]
            if "success_rate_mean" in eval["custom_metrics"]:
                result["success"] = eval["custom_metrics"]["success_rate_mean"]
                result["crash"] = eval["custom_metrics"]["crash_rate_mean"]
                result["out"] = eval["custom_metrics"]["out_of_road_rate_mean"]
                result["max_step"] = eval["custom_metrics"]["max_step_rate_mean"]
                result["native_cost"] = eval["custom_metrics"]["total_native_cost_mean"]
            if "cost_mean" in eval["custom_metrics"]:
                result["cost"] = eval["custom_metrics"]["cost_mean"]
            if "takeover_count_mean" in eval["custom_metrics"]:
                result["takeover"] = eval['custom_metrics']["takeover_count_mean"]
            if "episode_reward_mean" in eval:
                result["episode_reward"] = eval["episode_reward_mean"]
                result["episode_reward_mean"] = eval["episode_reward_mean"]
                result["reward"] = eval["episode_reward_mean"]
                result["length"] = eval["episode_len_mean"]


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def expert_action_prob(action, obs, weights, deterministic=False):
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    a_0_p = normpdf(action[0], mean[0], std[0])
    a_1_p = normpdf(action[1], mean[1], std[1])
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action, a_0_p, a_1_p


def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    # try:
    model = np.load(path)
    return model
    # except FileNotFoundError:
    # print("Can not find {}, didn't load anything".format(path))
    # return None


class CQLInputReader(InputReader):

    def __init__(self, data_set_path=None):
        super(CQLInputReader, self).__init__()
        assert data_set_path is not None
        with open(data_set_path, "r") as f:
            self.data = json.load(f)
        self.data_len = len(self.data)
        np.random.shuffle(self.data)
        self.count = 0

    def next(self) -> SampleBatch:
        if self.count == self.data_len:
            np.random.shuffle(self.data)
            self.count = 0
        index = self.count
        dp = self.data[index]
        # o,a,d,r,i
        batch = SampleBatch({SampleBatch.OBS: [dp[SampleBatch.OBS]],
                             SampleBatch.ACTIONS: [dp[SampleBatch.ACTIONS]],
                             SampleBatch.DONES: [dp[SampleBatch.DONES]],
                             SampleBatch.REWARDS: [dp[SampleBatch.REWARDS]],
                             SampleBatch.NEXT_OBS: [dp[SampleBatch.NEXT_OBS]],
                             # SampleBatch.INFOS: [dp[SampleBatch.INFOS]]
                             })
        self.count += 1
        return batch
    
    
class BCInputReader(InputReader):

    def __init__(self, data_set_path=None, max_expert_steps=10000):
        super(BCInputReader, self).__init__()
        assert data_set_path is not None
        self.expert_data_path = data_set_path
        import os
        self.rollouts_files = os.listdir(data_set_path)
        
        # obs = []
        # acts = []
        # new_obs = []
        # rewards = []
        self.data = []
        for rollout_dir in tqdm.tqdm(rollouts_files):
            if len(obs) >= max_expert_steps:
                break
            for t in range(len(os.listdir(os.path.join(expert_data_path, rollout_dir))) - 1):
                step_info = pickle.load(open(os.path.join(expert_data_path, rollout_dir, f'{t:05d}.pkl'), 'rb'))
                obs_t = step_info['obs']['state']
                action_info = pickle.load(open(os.path.join(expert_data_path, rollout_dir, f'{t + 1:05d}.pkl'), 'rb'))
                acts_t = action_info['info']['action']
                obs_nex = action_info['obs']['state']
                reward = action_info['reward']
                # obs.append(obs_t)
                # acts.append(acts_t)
                # rewards.append(reward)
                # new_obs.append(obs_nex)
                self.data.append([obs_t, acts_t, reward, obs_nex])
        
        self.data_len = len(obs)
        np.random.shuffle(self.data)
        self.count = 0

    def next(self) -> SampleBatch:
        if self.count == self.data_len:
            np.random.shuffle(self.data)
            self.count = 0
        index = self.count
        dp = self.data[index]
        # o,a,d,r,i
        batch = SampleBatch({SampleBatch.OBS: [dp[0]],
                             SampleBatch.ACTIONS: [dp[1]],
                             SampleBatch.REWARDS: [dp[2]],
                             SampleBatch.NEXT_OBS: [dp[3]],
                             # SampleBatch.INFOS: [dp[SampleBatch.INFOS]]
                             })
        self.count += 1
        return batch


class StateObservation(ObservationBase):
    def __init__(self, config):
        super(StateObservation, self).__init__(config)

    @property
    def observation_space(self):
        # Navi info + Other states
        shape = 19
        return gym.spaces.Box(-0.0, 1.0, shape=(shape,), dtype=np.float32)

    def observe(self, vehicle):
        navi_info = vehicle.navigation.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        ret = np.concatenate([ego_state, navi_info])
        return ret.astype(np.float32)

    def vehicle_state(self, vehicle):
        # update out of road
        current_reference_lane = vehicle.navigation.current_ref_lanes[-1]
        lateral_to_left, lateral_to_right = vehicle.dist_to_left_side, vehicle.dist_to_right_side
        total_width = float(
            (vehicle.navigation.map.config["lane_num"] + 1) * vehicle.navigation.map.config["lane_width"]
        )
        info = [
            clip(lateral_to_left / total_width, 0.0, 1.0),
            clip(lateral_to_right / total_width, 0.0, 1.0),
            vehicle.heading_diff(current_reference_lane),
            # Note: speed can be negative denoting free fall. This happen when emergency brake.
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),
            clip((vehicle.steering / vehicle.max_steering + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last
                                       ) / (np.linalg.norm(heading_dir_now) * np.linalg.norm(heading_dir_last))

        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))

        # print(beta)
        yaw_rate = beta_diff / 0.1
        # print(yaw_rate)
        info.append(clip(yaw_rate, 0.0, 1.0))
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        info.append(clip((lateral * 2 / vehicle.navigation.map.config["lane_width"] + 1.0) / 2.0, 0.0, 1.0))
        return info


class ExpertObservation(ObservationBase):
    def __init__(self, vehicle_config):
        self.state_obs = StateObservation(vehicle_config)
        super(ExpertObservation, self).__init__(vehicle_config)
        self.cloud_points = None
        self.detected_objects = None

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            shape[0] += self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
        ret = self.current_observation
        return ret.astype(np.float32)

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def lidar_observe(self, vehicle):
        other_v_info = []
        if vehicle.lidar.available:
            cloud_points, detected_objects = vehicle.lidar.perceive(vehicle, )
            other_v_info += vehicle.lidar.get_surrounding_vehicles_info(
                vehicle, detected_objects, 4)
            other_v_info += cloud_points
            self.cloud_points = cloud_points
            self.detected_objects = detected_objects
        return other_v_info


def get_expert_action(env):
    if not isinstance(env, SubprocVecEnv):
        obs = env.expert_observation.observe(env.vehicle)
        saver_a, a_0_p, a_1_p = expert_action_prob([0, 0], obs, env.expert_weights,
                                                   deterministic=False)
        return saver_a
    else:
        return env.env_method("get_expert_action")