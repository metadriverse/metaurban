# #
# # from metadrive.component.agents.pedestrian.base_pedestrian import BasePedestrian
# # from metadrive.component.pg_space import ParameterSpace, VehicleParameterSpace
# #
# #
# # class SimplePedestrian(BasePedestrian):
# #     PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
# #     # LENGTH = 1.
# #     # WIDTH = 1.85
# #     # HEIGHT = 1.37
# #     RADIUS = 0.35
# #
# #     # REAR_WHEELBASE = 1.203
# #     # FRONT_WHEELBASE = 1.285
# #     # LATERAL_TIRE_TO_CENTER = 0.803
# #     # TIRE_WIDTH = 0.3
# #     MASS = 80
# #     # LIGHT_POSITION = (-0.67, 1.86, 0.22)
# #
# #     # path = ['130/vehicle.gltf', (1, 1, 1), (0, -0.05, 0.1), (0, 0, 0)]
# #
# #     @property
# #     def LENGTH(self):
# #         return 1.  # meters
# #
# #     @property
# #     def HEIGHT(self):
# #         return 1.37  # meters
# #
# #     @property
# #     def WIDTH(self):
# #         return 1.  # meters

# from metadrive.component.agents.pedestrian.base_pedestrian import BasePedestrian
# from metadrive.component.pg_space import ParameterSpace, VehicleParameterSpace
# from metadrive.constants import AssetPaths
# from metadrive.utils.config import Config

# class SimplePedestrian(BasePedestrian):
#     PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

#     RADIUS = 0.35
#     MASS = 80

#     # def __init__(self, vehicle_config: dict | Config = None, name: str = None, random_seed=None, position=None, heading=None, _calling_reset=True):
#     #     super().__init__(vehicle_config, name, random_seed, position, heading, _calling_reset)

#     #     self.random_actor = AssetPaths.Pedestrian.get_random_actor()

#     @property
#     def LENGTH(self):
#         return 1.  # meters

#     @property
#     def HEIGHT(self):
#         if not hasattr(self, 'random_actor'):
#             self.random_actor = AssetPaths.Pedestrian.get_random_actor()
#         return self.random_actor['height']

#     @property
#     def WIDTH(self):
#         return 1.  # meters

#     @property
#     def ACTOR_PATH(self):
#         if not hasattr(self, 'random_actor'):
#             self.random_actor = AssetPaths.Pedestrian.get_random_actor()
#         return self.random_actor['actor_path']

#     @property
#     def MOTION_PATH(self):
#         if not hasattr(self, 'random_actor'):
#             self.random_actor = AssetPaths.Pedestrian.get_random_actor()
#         return self.random_actor['motion_path']

from metadrive.component.agents.pedestrian.base_pedestrian import BasePedestrian
from metadrive.component.pg_space import ParameterSpace, VehicleParameterSpace
from metadrive.constants import AssetPaths
from metadrive.utils.config import Config


class SimplePedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.01  #0.35
    MASS = 1000  # 80

    @property
    def LENGTH(self):
        return 1.  # meters

    # @property
    # def MAX_ACTOR_NUM(self):
    #     return self.engine.global_config.max_actor_num

    @property
    def HEIGHT(self):
        if not hasattr(self, 'random_actor'):
            if self.engine.global_config['use_fixed_walking_traj']:
                demo_ver = 2
            else:
                demo_ver = 0
            self.random_actor = AssetPaths.Pedestrian.get_random_actor(demo_ver=demo_ver)
            # else:
            #     self.random_actor = AssetPaths.Pedestrian.get_random_actor() #self.MAX_ACTOR_NUM)
        return self.random_actor['height']

    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()  #self.MAX_ACTOR_NUM)
        return self.random_actor['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()  #self.MAX_ACTOR_NUM)
        return self.random_actor['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()  #self.MAX_ACTOR_NUM)
        return 0 if 'actor_pitch' not in self.random_actor else self.random_actor['actor_pitch']

    @property
    def ACTOR_YAW(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()
        return 180 if 'actor_yaw' not in self.random_actor else self.random_actor['actor_yaw']

    @property
    def ACTOR_ROLL(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()
        return 0 if 'actor_roll' not in self.random_actor else self.random_actor['actor_roll']


class StaticPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.01  #0.35
    MASS = 1000  # 80

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'random_static_actor'):
            if self.engine.global_config['vis_bedlam_motion']:
                demo_ver = 1
            elif self.engine.global_config['fixed_standing_demo']:
                demo_ver = 3
            elif self.engine.global_config['use_customized_camera_traj']:
                demo_ver = 4
            else:
                demo_ver = 0
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor(
                demo_ver=demo_ver
            )  #self.MAX_ACTOR_NUM)
        return self.random_static_actor['height']

    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()  #self.MAX_ACTOR_NUM)
        return self.random_static_actor['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()  #self.MAX_ACTOR_NUM)
        return self.random_static_actor['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()  #self.MAX_ACTOR_NUM)
        return 0 if 'actor_pitch' not in self.random_static_actor else self.random_static_actor['actor_pitch']

    @property
    def ACTOR_YAW(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()
        return 180 if 'actor_yaw' not in self.random_static_actor else self.random_static_actor['actor_yaw']

    @property
    def ACTOR_ROLL(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()
        return 0 if 'actor_roll' not in self.random_static_actor else self.random_static_actor['actor_roll']


class EdogPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.01  #0.35
    MASS = 1000  # 80

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'edog_agent'):
            if self.engine.global_config['use_fixed_walking_traj']:
                demo_ver = 2
            else:
                demo_ver = 0
            self.edog_agent = AssetPaths.Pedestrian.get_edog_agent(demo_ver=demo_ver)
            # else:
            # self.edog_agent = AssetPaths.Pedestrian.get_edog_agent() #
        return self.edog_agent['height']

    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'edog_agent'):
            self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return self.edog_agent['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'edog_agent'):
            self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return self.edog_agent['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'edog_agent'):
            self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return 0 if 'actor_pitch' not in self.edog_agent else self.edog_agent['actor_pitch']

    @property
    def ACTOR_YAW(self):
        if not hasattr(self, 'edog_agent'):
            self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return 180 if 'actor_yaw' not in self.edog_agent else self.edog_agent['actor_yaw']

    @property
    def ACTOR_ROLL(self):
        if not hasattr(self, 'edog_agent'):
            self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return 0 if 'actor_roll' not in self.edog_agent else self.edog_agent['actor_roll']


class ErobotPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.01  #0.35
    MASS = 1000  # 80

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'erobot_agent'):
            if self.engine.global_config['use_fixed_walking_traj']:
                demo_ver = 2
            else:
                demo_ver = 0
            self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent(demo_ver=demo_ver)
            # else:
            #     self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent() #
        return self.erobot_agent['height']

    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'erobot_agent'):
            self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return self.erobot_agent['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'erobot_agent'):
            self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return self.erobot_agent['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'erobot_agent'):
            self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return 0 if 'actor_pitch' not in self.erobot_agent else self.erobot_agent['actor_pitch']

    @property
    def ACTOR_YAW(self):
        if not hasattr(self, 'erobot_agent'):
            self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return 180 if 'actor_yaw' not in self.erobot_agent else self.erobot_agent['actor_yaw']

    @property
    def ACTOR_ROLL(self):
        if not hasattr(self, 'erobot_agent'):
            self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return 0 if 'actor_roll' not in self.erobot_agent else self.erobot_agent['actor_roll']


class WheelchairPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.01  # 0.35
    MASS = 800  # 80

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'wheelchair_agent'):
            if self.engine.global_config['use_fixed_walking_traj']:
                demo_ver = 2
            else:
                demo_ver = 0
            self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent(demo_ver=demo_ver)
            # else:
            #     self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent() #
        return self.wheelchair_agent['height']

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'wheelchair_agent'):
            self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return self.wheelchair_agent['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'wheelchair_agent'):
            self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return self.wheelchair_agent['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'wheelchair_agent'):
            self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return 0 if 'actor_pitch' not in self.wheelchair_agent else self.wheelchair_agent['actor_pitch']

    @property
    def ACTOR_YAW(self):
        if not hasattr(self, 'wheelchair_agent'):
            self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return 180 if 'actor_yaw' not in self.wheelchair_agent else self.wheelchair_agent['actor_yaw']

    @property
    def ACTOR_ROLL(self):
        if not hasattr(self, 'wheelchair_agent'):
            self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return 0 if 'actor_roll' not in self.wheelchair_agent else self.wheelchair_agent['actor_roll']
