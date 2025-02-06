from metaurban.utils.math import wrap_to_pi

import copy
from metaurban.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metaurban.component.navigation_module.orca_navigation import ORCATrajectoryNavigation
from typing import Union

import numpy as np

from metaurban.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metaurban.component.map.base_map import BaseMap
from metaurban.component.map.pg_map import parse_map_config, MapGenerateMethod
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.constants import DEFAULT_AGENT, TerminationState
from metaurban.envs.base_env import BaseEnv
from metaurban.manager.traffic_manager import TrafficMode
from metaurban.utils import clip, Config

METAURBAN_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,

    # ===== PG Map Config =====
    map=3,  # int or string: an easy way to fill map_config
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
    },
    store_map=True,
    crswalk_density=0.1,  #####
    spawn_human_num=1,
    show_mid_block_map=False,
    # ===== Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block
    static_traffic_object=True,  # object won't react to any collisions

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,

    # ===== Agent =====
    random_spawn_lane_index=True,
    vehicle_config=dict(navigation_module=NodeNetworkNavigation, ego_navigation_module=ORCATrajectoryNavigation),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    # See: https://github.com/metaurbanrse/metaurban/issues/283
    success_reward=5.0,
    out_of_road_penalty=5.0,
    on_lane_line_penalty=1.,
    crash_vehicle_penalty=1.,
    crash_object_penalty=1.0,
    crash_human_penalty=1.0,
    driving_reward=1.0,
    steering_range_penalty=0.5,
    heading_penalty=1.0,
    lateral_penalty=.5,
    max_lateral_dist=2,
    no_negative_reward=True,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,
    crash_human_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    crash_human_done=False,
    relax_out_of_road_done=True,
)


class SidewalkDynamicMetaUrbanEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(SidewalkDynamicMetaUrbanEnv, cls).default_config()
        config.update(METAURBAN_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(SidewalkDynamicMetaUrbanEnv, self).__init__(config)

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios

        # record previous agent state
        self.previous_agent_actions = {}

    def _post_process_config(self, config):
        config = super(SidewalkDynamicMetaUrbanEnv, self)._post_process_config(config)
        if not config["norm_pixel"]:
            self.logger.warning(
                "You have set norm_pixel = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )

        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle) and not self._is_out_of_road(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        # determine env return
        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info['cost'], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        # Use RC as the only criterion to determine arrival in Scenario env.
        route_completion = vehicle.navigation.route_completion
        if route_completion > 0.95 or vehicle.navigation.reference_trajectory.length < 2:
            # Route Completion ~= 1.0 or vehicle is static!
            return True
        else:
            return False

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["relax_out_of_road_done"]:
            # We prefer using this out of road termination criterion.
            lat = abs(vehicle.navigation.current_lateral)
            done = lat > self.config["max_lateral_dist"]
            return done

    def record_previous_agent_state(self, vehicle_id: str):
        self.previous_agent_actions[vehicle_id] = self.agents[vehicle_id].current_action

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """

        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        current_lane = vehicle.lane
        long_last = vehicle.navigation.last_longitude
        long_now = vehicle.navigation.current_longitude
        lateral_now = vehicle.navigation.current_lateral

        # dense driving reward
        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last)

        # print('Long:', long_last, long_now)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        lateral_factor = abs(lateral_now) / self.config["max_lateral_dist"]
        lateral_penalty = -lateral_factor * self.config["lateral_penalty"]
        reward += lateral_penalty

        # heading diff
        ref_line_heading = vehicle.navigation.current_heading_theta_at_long
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi
        heading_penalty = -heading_diff * self.config["heading_penalty"]
        reward += heading_penalty

        # TODO: maybe add throttle smoothness

        # steering_range
        steering = abs(vehicle.current_action[0])
        allowed_steering = (1 / max(vehicle.speed, 1e-2))
        overflowed_steering = min((allowed_steering - steering), 0)
        steering_range_penalty = overflowed_steering * self.config["steering_range_penalty"]
        reward += steering_range_penalty

        # steering smoothness
        if vehicle_id not in self.previous_agent_actions or "steering_penalty" not in self.config or self.config[
                "steering_penalty"] == 0:
            steering_reward = 0
        else:
            steering = vehicle.current_action[0]
            prev_steering = self.previous_agent_actions[vehicle_id][0]
            steering_diff = abs(steering - prev_steering)
            steering_reward = -steering_diff * self.config["steering_penalty"]
        reward += steering_reward

        # if 'speed_reward' in self.config:
        #     positive_road = 1 if not self._is_out_of_road(vehicle) else -1
        #     reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        if self.config["no_negative_reward"]:
            reward = max(reward, 0)

        # crash penalty
        if vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        if vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        if vehicle.crash_human:
            reward = -self.config["crash_human_penalty"]
        # lane line penalty
        # if vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk or vehicle.on_white_continuous_line:
        #     reward = -self.config["on_lane_line_penalty"]

        step_info["step_reward"] = reward

        # termination reward
        if self._is_arrive_destination(vehicle) and not self._is_out_of_road(vehicle):
            reward = self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]

        # TODO LQY: all a callback to process these keys
        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["carsize"] = [vehicle.WIDTH, vehicle.LENGTH]
        # add some new and informative keys
        step_info["route_completion"] = vehicle.navigation.route_completion
        step_info["curriculum_level"] = self.engine.current_level
        step_info["scenario_index"] = self.engine.current_seed
        step_info["lateral_dist"] = lateral_now

        step_info["step_reward_lateral"] = lateral_penalty
        step_info["step_reward_heading"] = heading_penalty
        step_info["step_reward_action_smooth"] = steering_range_penalty
        step_info["steering_reward"] = steering_reward

        self.record_previous_agent_state(vehicle_id)
        return float(reward), step_info

    def setup_engine(self):
        super(SidewalkDynamicMetaUrbanEnv, self).setup_engine()
        from metaurban.manager.traffic_manager import NewAssetPGTrafficManager
        from metaurban.manager.humanoid_manager import PGBackgroundSidewalkAssetsManager as PGHumanoidManager
        from metaurban.manager.pg_map_manager import PGMapManager
        from metaurban.manager.object_manager import TrafficObjectManager
        from metaurban.manager.sidewalk_manager import AssetManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("asset_manager", AssetManager())
        self.engine.register_manager("traffic_manager", NewAssetPGTrafficManager())
        self.engine.register_manager("humanoid_manager", PGHumanoidManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())

    def _get_agent_manager(self):
        if 'agent_type' not in self.config:
            self.config['agent_type'] = 'coco'
        if self.config['agent_type'] == 'coco':
            from metaurban.manager.agent_manager import DeliveryRobotAgentManager
            return DeliveryRobotAgentManager(init_observations=self._get_observations())
        elif self.config['agent_type'] == 'wheelchair':
            from metaurban.manager.agent_manager import WheelchairAgentManager
            return WheelchairAgentManager(init_observations=self._get_observations())


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = SidewalkDynamicMetaUrbanEnv()
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
