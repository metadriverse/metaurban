import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
import tqdm

import numpy as np
import torch
import torch.nn as nn

from IL.utils.GAIL.exp_saver import Experiment
from IL.utils.GAIL.mlp import Policy, Value

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from easydict import EasyDict
from metaurban import SidewalkStaticMetaDriveEnv
from metaurban.utils import get_np_random
from metaurban.obs.state_obs import LidarStateObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# setup
exp_log = Experiment()
BACKBONE = 'resnet18'
dtype = torch.float32
torch.set_default_dtype(dtype)
N_STEP = 5

expert_data_path = ''
rollouts_files = os.listdir(expert_data_path)# TODO: argparse

# train env config
training_env_config = dict(
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
)


# eval env config
import copy
evaluation_config = copy.deepcopy(training_env_config)
eval_env_cfg = evaluation_config["env"]
eval_env_cfg["random_spawn"] = True
eval_env_cfg["horizon"] = 1000
eval_env_cfg.update(
    dict(crash_vehicle_done=True,
        crash_object_done=True,)
)


# make env fn
def make_metaurban_env_fn(env_cls, rank, config, seed=0):
    def _init():
        env = env_cls(config)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


# Learner
class Learner:
    def __init__(self, cfg: EasyDict):
        # self._init_cfg(cfg)
        # self._process_cfg()
        self.cfg = cfg
        self.env_num = 20

        # hyper para
        self.g_optim_num = 5
        self.d_optim_num = 2000
        self.sgd_batch_size = 64
        self.ppo_iterations = 200
        self.g_learning_rate = 1e-4
        self.d_learning_rate = 5e-3  # 1e-2
        self.eval_interval = 5
        self.eval_episodes = 10
        self.clip_epsilon = 0.2

        # auto calculate
        self.ppo_train_batch_size = self.sgd_batch_size * self.ppo_iterations
        self.buffer_length = int(self.sgd_batch_size * self.ppo_iterations / self.env_num)
        self.buffer = None
        self._init_env()
        self._load_expert_traj()
        tm_stamp = "%s-%s-%s-%s-%s-%s" % (tm.tm_year, tm.tm_mon, tm.tm_mday, \
                                          tm.tm_hour, tm.tm_min, tm.tm_sec)
        self.cfg.log_dir = os.path.join(
            "gail_pointnav_iter_{}_g_{}_d_{}_bs_{}_lr_d_{}_max_expert_steps_{}".format(self.ppo_iterations, self.g_optim_num, self.d_optim_num,
                                                          self.sgd_batch_size, self.d_learning_rate, self.cfg.max_expert_steps),
            tm_stamp)
        self.policy_net = Policy(state_dim=271, action_dim=2).to(self.cfg.device).float()
        self.value_net = Value(state_dim=271 + 2).to(self.cfg.device).float()
        self.eval_env = SidewalkStaticMetaDriveEnv(eval_env_cfg)

    def _load_expert_traj(self):
        import pickle
        try:
            obs = []
            acts = []
            for rollout_dir in tqdm.tqdm(rollouts_files):
                if len(obs) >= self.cfg.max_expert_steps:
                    break
                for t in range(len(os.listdir(os.path.join(expert_data_path, rollout_dir))) - 1):
                    step_info = pickle.load(open(os.path.join(expert_data_path, rollout_dir, f'{t:05d}.pkl'), 'rb'))
                    obs_t = step_info['obs']['state']
                    action_info = pickle.load(open(os.path.join(expert_data_path, rollout_dir, f'{t + 1:05d}.pkl'), 'rb'))
                    acts_t = action_info['info']['action']
                    obs.append(obs_t)
                    acts.append(acts_t)

            self.exp_obs = torch.tensor(obs).to(self.cfg.device).float()
            self.exp_action = torch.tensor(acts).to(self.cfg.device).float()
        except FileNotFoundError:
            raise ValueError("Please collect dataset by using collect_dataset.py at first")

    def _init_env(self):
        # self.env = PGDriveEnv(dict(environment_num=1))
        self.env = SubprocVecEnv(
            [make_metaurban_env_fn(SidewalkStaticMetaDriveEnv, i, config=training_env_config['env']) for i in range(self.env_num)])
        # self.env = make_vec_env('PGDrive-v0', n_envs=self.env_num, seed=0)

    def _process_cfg(self):
        if isinstance(self.cfg.get('device', torch.device('cpu')), str):
            assert self.cfg.device in ['cpu', 'cuda']
            self.cfg.device = torch.device(self.cfg.device)

    def _collect_samples(self):
        obs = self.env.reset()
        batch_obs = []
        batch_prob = []
        batch_action = []
        batch_reward = []

        # training metric
        done_num = 1
        success_num = 0
        episode_reward_mean = [0 for _ in range(self.env_num)]
        episode_cost_mean = [0 for _ in range(self.env_num)]
        total_episode_reward = 0
        total_episode_cost = 0
        for i in range(self.buffer_length):
            obs = torch.tensor(obs).to(self.cfg.device).float()
            with torch.no_grad():
                action, prob = self.policy_net.select_action(obs)
            batch_obs.append(obs)
            batch_prob.append(prob)
            batch_action.append(action)
            obs, reward, dones, info = self.env.step(action.cpu().numpy())
            batch_reward.append(torch.tensor(reward))

            episode_reward_mean = [episode_reward_mean[i] + reward[i] for i in range(self.env_num)]
            episode_cost_mean = [episode_cost_mean[i] + info[i]["cost"] for i in range(self.env_num)]

            # asyn done
            for idx, done in enumerate(dones):
                if done:
                    done_num += 1
                    success_num += 1 if info[idx]["arrive_dest"] else 0
                    total_episode_reward += episode_reward_mean[idx]
                    total_episode_cost += episode_cost_mean[idx]
                    episode_reward_mean[idx] = 0
                    episode_cost_mean[idx] = 0
                    self.env.remotes[idx].send(("reset", None))
                    this_obs, _ = self.env.remotes[idx].recv()
                    obs[idx] = this_obs
        # return data
        perm = np.arange(self.buffer_length * self.env_num)
        np.random.shuffle(perm)
        self.buffer = OrderedDict({
            'obs': torch.cat(batch_obs, 0)[perm],
            'action': torch.cat(batch_action, 0)[perm],
            'prob': torch.cat(batch_prob, 0)[perm],
            'reward': torch.cat(batch_reward)[perm],
        })
        return {"episode_reward_mean": total_episode_reward / done_num,
                "success_rate_mean": success_num / done_num,
                "episode_cost_mean": total_episode_cost / done_num,}

    def _sample_from_buffer(self, batch_size, cnt):
        start = batch_size * cnt
        end = min(batch_size * (cnt + 1), self.buffer_length * self.env_num)
        return (v[start:end] for k, v in self.buffer.items())

    def evaluation(self, evaluation_episode_num=30):
        env = self.eval_env
        print("... evaluation")
        episode_reward = 0
        success_num = 0
        episode_num = 0
        episode_cost = 0
        velocity = []
        state, _ = env.reset()
        episode_overtake = []
        while episode_num < evaluation_episode_num:
            state = torch.tensor([state]).to(self.cfg.device).float()
            with torch.no_grad():
                action, prob = self.policy_net.select_action(state)
            next_state, r, done, _, info = env.step(action.cpu().numpy()[0])
            state = next_state
            episode_reward += r
            episode_cost += info["cost"]
            if done:
                print('Evalution Episode Done, current:', episode_num)
                episode_num += 1
                env.reset()
                if info["arrive_dest"]:
                    success_num += 1
        res = dict(
            mean_episode_reward=episode_reward / evaluation_episode_num,
            mean_success_rate=success_num / evaluation_episode_num,
            mean_episode_cost=episode_cost / evaluation_episode_num,
        )
        return res

    def train(self, is_train):
        self.policy_net.train()
        self.value_net.train()
        tick = time.time()
        sample_result = self._collect_samples()

        # train discriminator
        d_loss_list = []
        obs = self.buffer["obs"]
        action = self.buffer["action"]
        for _ in range(self.d_optim_num):
            g_o = self.value_net(torch.cat([obs, action], 1))
            e_o = self.value_net(torch.cat([self.exp_obs, self.exp_action], 1))
            discrim_loss = nn.BCELoss().float()(g_o, torch.zeros((obs.shape[0], 1)).cuda()) + \
                           nn.BCELoss().float()(e_o, torch.ones((self.exp_obs.shape[0], 1)).cuda())
            with torch.no_grad():
                d_loss_list.append(discrim_loss.item())
            # update d
            self.optim_d.zero_grad()
            discrim_loss.backward()
            self.optim_d.step()
        d_loss_mean = sum(d_loss_list) / len(d_loss_list)

        # train generator
        rl_loss_list = []
        step_reward = []
        for opt_idx in range(self.g_optim_num):
            for i in range(self.ppo_iterations):
                # obs, action, prob = self._collect_samples()
                obs, action, prob, real_reward = self._sample_from_buffer(self.sgd_batch_size, i)
                step_reward += real_reward
                obs = obs.to(self.cfg.device).float()
                action = action.to(self.cfg.device).float()
                prob = prob.to(self.cfg.device).float()
                g_o = self.value_net(torch.cat([obs, action], 1))

                # update g
                reward = g_o.detach()
                obs_s, action_s, log_p_old_s, reward_s = obs, action, prob, reward

                # perform ppo step
                log_p = self.policy_net.get_log_prob(obs_s, action_s)
                ratio = (log_p - log_p_old_s).exp().float()
                surr1 = ratio * reward_s
                surr2 = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * reward_s
                rl_loss = -torch.min(surr1, surr2).mean()
                rl_loss_list.append(rl_loss.item())

                self.optim_g.zero_grad()
                rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy_net.parameters()), 10)
                self.optim_g.step()
        loss_mean = sum(rl_loss_list) / len(rl_loss_list)

        metrics = dict()
        metrics['generator_loss'] = loss_mean
        metrics['discriminator_loss'] = d_loss_mean
        metrics['step_reward'] = np.mean(step_reward)
        metrics["episode_reward"] = sample_result["episode_reward_mean"]
        metrics["episode_cost"] = sample_result["episode_cost_mean"]
        metrics["success_rate"] = sample_result["success_rate_mean"]

        exp_log.scalar(is_train=is_train, **metrics)
        exp_log.scalar(is_train=is_train, fps=self.sgd_batch_size * self.ppo_iterations / (time.time() - tick))

    def learn(self):
        exp_log.init(self.cfg.log_dir)
        if self.cfg.resume:
            log_dir = Path(self.cfg.resume_dir)
            checkpoints_d = list(log_dir.glob('model_d_*.th'))
            checkpoints_g = list(log_dir.glob('model_g_*.th'))
            checkpoint_d = str(checkpoints_d[-1])
            checkpoint_g = str(checkpoints_g[-1])
            print("load {} {}".format(checkpoint_d, checkpoint_g))
            self.policy_net.load_state_dict(torch.load(checkpoint_g))
            self.value_net.load_state_dict(torch.load(checkpoint_d))

        self.optim_d = torch.optim.Adam(self.value_net.parameters(), lr=self.d_learning_rate)
        self.optim_g = torch.optim.Adam(self.policy_net.parameters(), lr=self.g_learning_rate)

        for epoch in range(self.cfg.max_epoch + 1):
            self.train(True)
            if epoch % self.cfg.save_freq == 0:
                torch.save(
                    self.value_net.state_dict(),
                    str(Path(self.cfg.log_dir) / ('model_d_%d.th' % epoch)))
                torch.save(
                    self.policy_net.state_dict(),
                    str(Path(self.cfg.log_dir) / ('model_g_%d.th' % epoch)))
            if epoch % self.eval_interval == 0:
                res = self.evaluation(self.eval_episodes)
                exp_log.scalar(is_train=False, **res)
            exp_log.end_epoch(epoch, net=self.policy_net)


if __name__ == '__main__':
    torch.set_default_dtype(dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--log_iterations', default=10)
    parser.add_argument('--max_epoch', default=100000)
    parser.add_argument('--save_freq', default=20)

    # Dataset.
    parser.add_argument('--dataset_dir', default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--cmd-biased', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--max_expert_steps', type=int, default=5000)

    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)

    parsed = parser.parse_args()
    cfg = EasyDict({
        'log_dir': parsed.log_dir + '_' + f'{parsed.max_expert_steps:09d}',
        'resume': parsed.resume,
        'log_iterations': parsed.log_iterations,
        'save_freq': parsed.save_freq,
        'max_epoch': parsed.max_epoch,
        'device': torch.device('cuda'),  # force use cuda
        'optimizer_args': {'lr': parsed.lr},
        'data_args': {
            'dataset_dir': parsed.dataset_dir,
            'batch_size': parsed.batch_size,
            'n_step': N_STEP,
            'max_frames': parsed.max_frames,
            'cmd_biased': parsed.cmd_biased,
        },
        'model_args': {
            'model': 'birdview_dian',
            'input_channel': 7,
            'backbone': BACKBONE,
        },
        'max_expert_steps': parsed.max_expert_steps,
        'resume_dir': parsed.resume_dir
    })
    tm = time.localtime(time.time())
    tm_stamp = "%s-%s-%s-%s-%s-%s" % (tm.tm_year, tm.tm_mon, tm.tm_mday, \
                                      tm.tm_hour, tm.tm_min, tm.tm_sec)
    cfg.log_dir = os.path.join(cfg.log_dir, tm_stamp)
    il_learner = Learner(cfg)
    il_learner.learn()
    