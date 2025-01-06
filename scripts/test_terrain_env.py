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
    map='S',  # int or string: an easy way to fill map_config
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


class TestTerrainEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(TestTerrainEnv, cls).default_config()
        config.update(METAURBAN_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(TestTerrainEnv, self).__init__(config)

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios
        
        # record previous agent state
        self.previous_agent_actions = {}

    def _post_process_config(self, config):
        config = super(TestTerrainEnv, self)._post_process_config(config)
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
        return False, {}

    def cost_function(self, vehicle_id: str):
        return 0., {}

    @staticmethod
    def _is_arrive_destination(vehicle):
        return False

    def _is_out_of_road(self, vehicle):
        return False

    def record_previous_agent_state(self, vehicle_id: str):
        self.previous_agent_actions[vehicle_id] = self.agents[vehicle_id].current_action
    
    def reward_function(self, vehicle_id: str):
        return 0., {}

    def setup_engine(self):
        super(TestTerrainEnv, self).setup_engine()
        from metaurban.manager.pg_map_manager import PGMapManager
        self.engine.register_manager("map_manager", PGMapManager())
            
    def _get_agent_manager(self):
        from metaurban.manager.agent_manager import DeliveryRobotAgentManager
        from metaurban.manager.agent_manager import VehicleAgentManager
        return DeliveryRobotAgentManager(init_observations=self._get_observations())


if __name__ == "__main__":
    map_type = 'S'
    config = dict(
        crswalk_density=1,
        object_density=0.8,
        use_render=True,
        map = map_type,
        manual_control=False,
        drivable_area_extension=55,
        height_scale = 1,
        spawn_deliveryrobot_num=2,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=True,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        # scenario setting
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        window_size=(960, 960),
        relax_out_of_road_done=True,
        max_lateral_dist=5.0,
        test_terrain_system=False,
        test_slope_system=False,
        test_rough_system=True,
    )
    
    env = TestTerrainEnv(config)
    o, _ = env.reset(seed=0)
    env.engine.toggleDebug()
    last_pos = [env.agents['default_agent'].position[0], env.agents['default_agent'].position[1]]
    moving_dis = 0.
    try:
        for i in range(1, 1000000000):
                
            o, r, tm, tc, info = env.step([0., 1.0])   ### reset; get next -> empty -> have multiple end points
            curr_pos = [env.agents['default_agent'].position[0], env.agents['default_agent'].position[1]]
            dis = np.sqrt((curr_pos[1] - last_pos[1]) ** 2 + (curr_pos[0] - last_pos[0]) ** 2 )
            last_pos = curr_pos
            moving_dis += dis
            print(i - 1, moving_dis)

            if (tm or tc):
                env.reset(env.current_seed + 1)
    finally:
        env.close()

