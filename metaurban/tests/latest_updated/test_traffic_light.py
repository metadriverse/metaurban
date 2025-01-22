from metaurban.component.traffic_light.base_traffic_light import BaseTrafficLight
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.policy.idm_policy import IDMPolicy
class VehicleSidewalkStaticMetaUrbanEnv(SidewalkStaticMetaUrbanEnv):
    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        from metaurban.utils import clip, Config
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info
    
    def _get_agent_manager(self):
        from metaurban.manager.agent_manager import VehicleAgentManager
        return VehicleAgentManager(init_observations=self._get_observations())


def test_traffic_light_idm_policy(render=False, debug=False):
    env = VehicleSidewalkStaticMetaUrbanEnv(
        {
            'use_lateral_reward': False,
            'object_density': 0.1,
            "num_scenarios": 1,
            "traffic_density": 0.,
            "traffic_mode": "hybrid",
            "agent_policy": IDMPolicy,
            "use_render": render,
            "debug": debug,
            "map": "X",
            "window_size": (1200, 800),
            "show_coordinates": True,
            "vehicle_config": {
                "show_lidar": True,
                "enable_reverse": True,
                "show_dest_mark": True
            },
            'speed_reward':0.1,
            'on_broken_line_done':False,
            'on_continuous_line_done':False,
        }
    )
    env.reset()
    try:
        # green
        env.reset()
        light = env.engine.spawn_object(BaseTrafficLight, lane=env.current_map.road_network.graph[">>>"]["1X1_0_"][0])
        light.set_green()
        for s in range(1, 200):
            if s == 30:
                light.set_yellow()
            elif s == 90:
                light.set_red()
            env.step([0, 1])
            if env.vehicle.red_light or env.vehicle.yellow_light:
                raise ValueError("Vehicle should not stop at red light!")

        # move
        light.set_green()
        test_success = False
        for s in range(1, 1000):
            o, r, d, t, i = env.step([0, 1])
            if i["arrive_dest"]:
                test_success = True
                break
        light.destroy()
        assert test_success
    finally:
        env.close()


if __name__ == "__main__":
    test_traffic_light_idm_policy(True)
    