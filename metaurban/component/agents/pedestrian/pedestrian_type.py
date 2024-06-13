# #
# # from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian
# # from metaurban.component.pg_space import ParameterSpace, VehicleParameterSpace
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




# from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian
# from metaurban.component.pg_space import ParameterSpace, VehicleParameterSpace
# from metaurban.constants import AssetPaths
# from metaurban.utils.config import Config


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


from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian
from metaurban.component.pg_space import ParameterSpace, VehicleParameterSpace
from metaurban.constants import AssetPaths
from metaurban.utils.config import Config


class SimplePedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.35
    MASS = 80

    @property
    def LENGTH(self):
        return 1.  # meters

    # @property
    # def MAX_ACTOR_NUM(self):
    #     return self.engine.global_config.max_actor_num
    
    @property
    def HEIGHT(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor() #self.MAX_ACTOR_NUM)
        return self.random_actor['height']


    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()#self.MAX_ACTOR_NUM)
        return self.random_actor['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()#self.MAX_ACTOR_NUM)
        return self.random_actor['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'random_actor'):
            self.random_actor = AssetPaths.Pedestrian.get_random_actor()#self.MAX_ACTOR_NUM)
        return 0 if 'actor_pitch' not in self.random_actor else self.random_actor['actor_pitch']
    

class StaticPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.35
    MASS = 80

    @property
    def LENGTH(self):
        return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor() #self.MAX_ACTOR_NUM)
        return self.random_static_actor['height']


    @property
    def WIDTH(self):
        return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()#self.MAX_ACTOR_NUM)
        return self.random_static_actor['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()#self.MAX_ACTOR_NUM)
        return self.random_static_actor['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'random_static_actor'):
            self.random_static_actor = AssetPaths.Pedestrian.get_static_random_actor()#self.MAX_ACTOR_NUM)
        return 0 if 'actor_pitch' not in self.random_static_actor else self.random_static_actor['actor_pitch']
    



class EdogPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.35
    MASS = 80

    @property
    def LENGTH(self): return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'edog_agent'): self.random_static_actor = AssetPaths.Pedestrian.get_edog_agent() #
        return self.random_static_actor['height']


    @property
    def WIDTH(self): return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'edog_agent'): self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return self.edog_agent['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'edog_agent'): self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return self.edog_agent['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'edog_agent'): self.edog_agent = AssetPaths.Pedestrian.get_edog_agent()
        return 0 if 'actor_pitch' not in self.edog_agent else self.edog_agent['actor_pitch']
    


class ErobotPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.35
    MASS = 80

    @property
    def LENGTH(self): return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'erobot_agent'): self.random_static_actor = AssetPaths.Pedestrian.get_erobot_agent() #
        return self.random_static_actor['height']


    @property
    def WIDTH(self): return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'erobot_agent'): self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return self.erobot_agent['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'erobot_agent'): self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return self.erobot_agent['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'erobot_agent'): self.erobot_agent = AssetPaths.Pedestrian.get_erobot_agent()
        return 0 if 'actor_pitch' not in self.erobot_agent else self.erobot_agent['actor_pitch']
    


class WheelchairPedestrian(BasePedestrian):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)

    RADIUS = 0.35
    MASS = 80

    @property
    def LENGTH(self): return 1.  # meters

    @property
    def HEIGHT(self):
        if not hasattr(self, 'wheelchair_agent'): self.random_static_actor = AssetPaths.Pedestrian.get_wheelchair_agent() #
        return self.random_static_actor['height']


    @property
    def WIDTH(self): return 1.  # meters

    @property
    def ACTOR_PATH(self):
        if not hasattr(self, 'wheelchair_agent'): self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return self.wheelchair_agent['actor_path']

    @property
    def MOTION_PATH(self):
        if not hasattr(self, 'wheelchair_agent'): self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return self.wheelchair_agent['motion_path']

    @property
    def ACTOR_PITCH(self):
        if not hasattr(self, 'wheelchair_agent'): self.wheelchair_agent = AssetPaths.Pedestrian.get_wheelchair_agent()
        return 0 if 'actor_pitch' not in self.wheelchair_agent else self.wheelchair_agent['actor_pitch']
    