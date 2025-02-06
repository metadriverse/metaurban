from direct.actor.Actor import Actor
from panda3d.bullet import BulletCylinderShape, BulletCapsuleShape

from metaurban.constants import MetaUrbanType, AssetPaths

import time

import math
import os
from collections import deque
from typing import Union, Optional

import numpy as np
import seaborn as sns
from panda3d._rplight import RPSpotLight
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp, BulletCharacterControllerNode
from panda3d.core import Material, Vec3, TransformState
from panda3d.core import NodePath
from panda3d.core import Point3, Vec2, LPoint3f, Material
from metaurban.engine.asset_loader import AssetLoader
from metaurban.base_class.base_object import BaseObject
from metaurban.component.lane.abs_lane import AbstractLane
from metaurban.component.lane.circular_lane import CircularLane
from metaurban.component.lane.point_lane import PointLane
from metaurban.component.lane.straight_lane import StraightLane
from metaurban.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metaurban.component.pg_space import VehicleParameterSpace, ParameterSpace
from metaurban.constants import CamMask
from metaurban.constants import MetaUrbanType, CollisionGroup
from metaurban.constants import Semantics
# from metaurban.engine.asset_loader import AssetLoader
from metaurban.engine.engine_utils import get_engine, engine_initialized
from metaurban.engine.logger import get_logger
# from metaurban.engine.physics_node import BaseRigidBodyNode
from metaurban.utils import Config, safe_clip_for_small_array
from metaurban.utils.math import get_vertical_vector, norm, clip
from metaurban.utils.math import wrap_to_pi
from metaurban.utils.pg.utils import rect_region_detection
from metaurban.utils.utils import get_object_from_node
from panda3d.core import BitMask32
from copy import deepcopy
import random
logger = get_logger()


class BaseDeliveryRobotState:
    def __init__(self):
        self.init_state_info()

    def init_state_info(self):
        """
        Call this before reset()/step()
        """
        self.crash_vehicle = False
        self.crash_human = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False

        # traffic light
        self.red_light = False
        self.yellow_light = False
        self.green_light = False

        # lane line detection
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False
        self.on_crosswalk = False

        # contact results, a set containing objects type name for rendering
        self.contact_results = set()


class BaseDeliveryRobot(BaseObject, BaseDeliveryRobotState):
    """

    """
    COLLISION_MASK = CollisionGroup.Vehicle
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
    MAX_LENGTH = 10
    MAX_WIDTH = 2.5
    MAX_STEERING = 60
    SEMANTIC_LABEL = Semantics.CAR.label

    # control
    # STEERING_INCREMENT = 0.05

    # save memory, load model once
    model_collection = {}
    # path = None
    STATES = ['walk', "run", "idle"]

    # TYPE_NAME = MetaUrbanType.PEDESTRIAN
    # velocity = (0, 0, 0)
    speed = 0
    steering = 0

    def __init__(
        self,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
        position=None,
        heading=None,
        _calling_reset=True,
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param vehicle_config: mostly, vehicle module config
        :param random_seed: int
        """
        # check
        assert vehicle_config is not None, "Please specify the vehicle config."
        assert engine_initialized(), "Please make sure game engine is successfully initialized!"

        # NOTE: it is the game engine, not vehicle drivetrain
        # self.engine = get_engine()
        BaseObject.__init__(self, name, random_seed, self.engine.global_config["vehicle_config"])
        BaseDeliveryRobotState.__init__(self)
        self.update_config(vehicle_config)
        self.set_metaurban_type(MetaUrbanType.VEHICLE)

        body = self._create_vehicle_chassis().getChassis()
        self.add_body(body)

        self.system = body
        self.chassis = self.origin

        # visualization
        self._add_visualization()
        # navigation module
        self.navigation: Optional[NodeNetworkNavigation] = None

        # state info
        # self.speed = 1 # env required, not used
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = (0, 0)
        self.last_heading_dir = self.heading
        self.dist_to_left_side = None
        self.dist_to_right_side = None
        self.last_velocity = 0
        self.last_speed = 0

        # step info
        self.out_of_route = None
        self.on_lane = None
        self.spawn_place = (0, 0)
        self._init_step_info()

        # others
        self.takeover = False
        self.expert_takeover = False
        self.energy_consumption = 0
        self.break_down = False

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        # if self.engine.current_map is not None:
        if _calling_reset:
            self.reset(position=position, heading=heading, vehicle_config=vehicle_config)

    def _init_step_info(self):
        # done info will be initialized every frame
        self.init_state_info()
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not

    @staticmethod
    def _preprocess_action(action):
        if action is None:
            return None, {"raw_action": None}
        action = safe_clip_for_small_array(action, -1, 1)
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        # if action is None:
        #     action = [0, 0]

        self._init_step_info()
        action, step_info = self._preprocess_action(action)

        self.last_position = self.position  # 2D vector
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar
        self.last_heading_dir = self.heading
        if action is not None:
            self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        # if self.increment_steering:
        #     self._set_incremental_action(action)
        # else:
        self._set_action(action)
        return step_info

    def after_step(self):
        if self.navigation and self.config["navigation_module"]:
            self.navigation.update_localization(self)
        self._state_check()
        self.update_dist_to_left_right()
        step_energy, episode_energy = self._update_energy_consumption()
        self.out_of_route = self._out_of_route()
        step_info = self._update_overtake_stat()
        my_policy = self.engine.get_policy(self.name)
        step_info.update(
            {
                "velocity": float(self.speed),
                "steering": float(self.steering),
                "acceleration": float(self.throttle_brake),
                "step_energy": step_energy,
                "episode_energy": episode_energy,
                "policy": my_policy.name if my_policy is not None else my_policy
            }
        )
        return step_info

    def _out_of_route(self):
        left, right = self._dist_to_route_left_right()
        return True if right < 0 or left < 0 else False

    def _update_energy_consumption(self):
        """
        The calculation method is from
        https://www.researchgate.net/publication/262182035_Reduction_of_Fuel_Consumption_and_Exhaust_Pollutant_Using_Intelligent_Transport_System
        default: 3rd gear, try to use ae^bx to fit it, dp: (90, 8), (130, 12)
        :return: None
        """
        distance = norm(self.last_position[0] - self.position[0], self.last_position[1] - self.position[1]) / 1000  # km
        # print("distance", distance / self.engine.global_config["physics_world_step_size"] * 1000)
        step_energy = 3.25 * math.pow(np.e, 0.01 * self.speed_km_h) * distance / 100
        # step_energy is in Liter, we return mL
        step_energy = step_energy * 1000
        self.energy_consumption += step_energy  # L/100 km
        return step_energy, self.energy_consumption

    def reset(
        self,
        vehicle_config=None,
        name=None,
        random_seed=None,
        position: np.ndarray = None,
        heading: float = 0.0,
        *args,
        **kwargs
    ):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if name is not None:
            self.rename(name)

        # reset fully
        self.update_config(self.engine.global_config["vehicle_config"])
        if random_seed is not None:
            assert isinstance(random_seed, int)
            self.seed(random_seed)
            self.sample_parameters()

        if vehicle_config is not None:
            self.update_config(vehicle_config)
        # from metaurban.component.vehicle.vehicle_type import vehicle_class_to_type
        # self.config["vehicle_model"] = vehicle_class_to_type[self.__class__]
        self.config["vehicle_model"] = "BasePedestrian"

        # Update some modules that might not be initialized before
        self.add_navigation()

        self.set_pitch(0)
        self.set_roll(0)
        if position is not None:
            # Highest priority
            pass
        elif self.config["spawn_position_heading"] is None:
            # spawn_lane_index has second priority
            map = self.engine.current_map
            if map is None:
                logger.warning("No map is provided. Set vehicle to position (0, 0) with heading 0")
                position = [0, 0]
                heading = 0
            else:
                lane = map.road_network.get_lane(self.config["spawn_lane_index"])
                position = lane.position(self.config["spawn_longitude"], self.config["spawn_lateral"])
                heading = lane.heading_theta_at(self.config["spawn_longitude"])
        else:
            assert self.config["spawn_position_heading"] is not None, "At least setting one initialization method"
            position = self.config["spawn_position_heading"][0]
            heading = self.config["spawn_position_heading"][1]

        self.spawn_place = position
        self.set_heading_theta(heading)
        """
        # self.set_static(False)
        """

        if len(position) == 2:
            self.set_position(position, height=self.HEIGHT / 2)
        elif len(position) == 3:
            self.set_position(position[:2], height=position[-1])
        else:
            raise ValueError()
        try:
            self.reset_navigation()
        except Exception:
            pass
            # print("error self.reset_navigation()")
        # self.body.setLinearMovement(Vec3(0, 0, 0), True)
        # self.body.setAngularMovement(0)

        # # self.system.resetSuspension()
        # # self._apply_throttle_brake(0.0)
        # # np.testing.assert_almost_equal(self.position, pos, decimal=4)

        # # done info
        # self._init_step_info()

        # # other info
        # self.throttle_brake = 0.0
        # self.steering = 0
        # self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        # self.last_position = self.spawn_place
        # self.last_heading_dir = self.heading
        # self.last_velocity = self.velocity  # 2D vector
        # self.last_speed = self.speed  # Scalar
        # try:
        #     self.update_dist_to_left_right()
        # except Exception:
        #     pass
        #     # print("error self.update_dist_to_left_right()")
        # self.takeover = False
        # self.energy_consumption = 0

        # # overtake_stat
        # self.front_vehicles = set()
        # self.back_vehicles = set()
        # self.expert_takeover = False
        # if self.config["navigation_module"] and self.engine.current_map is not None:
        #     assert self.navigation

        # if self.config["spawn_velocity"] is not None:
        #     self.set_velocity(self.config["spawn_velocity"], in_local_frame=self.config["spawn_velocity_car_frame"])

        self.body.clearForces()
        self.body.setLinearVelocity(Vec3(0, 0, 0))
        self.body.setAngularVelocity(Vec3(0, 0, 0))
        # np.testing.assert_almost_equal(self.position, pos, decimal=4)

        # done info
        self._init_step_info()

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.spawn_place
        self.last_heading_dir = self.heading
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar

        self.takeover = False
        self.energy_consumption = 0

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()
        self.expert_takeover = False
        if self.config["ego_navigation_module"] and self.engine.current_map is not None:
            assert self.navigation

        if self.config["spawn_velocity"] is not None:
            self.set_velocity(self.config["spawn_velocity"], in_local_frame=self.config["spawn_velocity_car_frame"])

    """------------------------------------------- act -------------------------------------------------"""

    def set_steering(self, steering):
        steering = float(steering)
        self.steering = steering

    def set_throttle_brake(self, throttle_brake):
        throttle_brake = float(throttle_brake)
        self._apply_throttle_brake(throttle_brake)
        self.throttle_brake = throttle_brake

    def _set_action(self, action):
        if action is None:
            return

        steering = action[0] * 90  # / np.pi * 180
        speed = action[1] * 10

        self._body.setAngularMovement(steering)
        self._body.setLinearMovement(LPoint3f(0, 1, 0) * speed, True)

    def _set_incremental_action(self, action: np.ndarray):
        raise NotImplementedError("_set_incremental_action")
        if action is None:
            return
        self.throttle_brake = action[1]
        self.steering += action[0] * self.STEERING_INCREMENT
        self.steering = clip(self.steering, -1, 1)
        steering = self.steering * self.max_steering
        # self.system.setSteeringValue(steering, 0)
        # self.system.setSteeringValue(steering, 1)
        self._apply_throttle_brake(action[1])

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def get_forward_vector(self):
        return get_engine().render.getRelativeVector(self.origin, Vec3(0, 1, 0))

    def update_dist_to_left_right(self):
        self.dist_to_left_side, self.dist_to_right_side = self._dist_to_route_left_right()

    def _dist_to_route_left_right(self):
        # TODO
        if self.navigation is None or self.navigation.current_ref_lanes is None:
            return 0, 0
        current_reference_lane = self.navigation.current_ref_lanes[0]
        _, lateral_to_reference = current_reference_lane.local_coordinates(self.position)
        lateral_to_left = lateral_to_reference + self.navigation.get_current_lane_width() / 2
        lateral_to_right = self.navigation.get_current_lateral_range(self.position, self.engine) - lateral_to_left
        return lateral_to_left, lateral_to_right

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * np.array([1, 1])

    @property
    def velocity_km_h(self) -> np.ndarray:
        return self.speed * np.array(self.get_forward_vector())[:2] * 3.6

    @property
    def speed_km_h(self):
        return self.speed * 3.6

    @property
    def speed(self):
        distance = norm(self.last_position[0] - self.position[0], self.last_position[1] - self.position[1]) / 1000  # km
        speed = distance / self.engine.global_config["physics_world_step_size"] * 1000

        return speed

    @property
    def chassis_velocity_direction(self):
        raise DeprecationWarning(
            "This API returns the direction of velocity which is approximately heading direction. "
            "Deprecate it and make things easy"
        )
        # direction = self.system.getForwardVector()
        # return np.asarray([direction[0], direction[1]])

    """---------------------------------------- some math tool ----------------------------------------------"""

    def heading_diff(self, target_lane):
        lateral = None
        if isinstance(target_lane, StraightLane):
            lateral = np.asarray(get_vertical_vector(target_lane.end - target_lane.start)[1])
        elif isinstance(target_lane, CircularLane):
            if not target_lane.is_clockwise():
                lateral = self.position - target_lane.center
            else:
                lateral = target_lane.center - self.position
        elif isinstance(target_lane, PointLane):
            lateral = target_lane.lateral_direction(target_lane.local_coordinates(self.position)[0])

        lateral_norm = norm(lateral[0], lateral[1])
        forward_direction = self.heading
        # print(f"Old forward direction: {self.forward_direction}, new heading {self.heading}")
        forward_direction_norm = norm(forward_direction[0], forward_direction[1])
        if not lateral_norm * forward_direction_norm:
            return 0
        cos = (
            (forward_direction[0] * lateral[0] + forward_direction[1] * lateral[1]) /
            (lateral_norm * forward_direction_norm)
        )
        # return cos
        # Normalize to 0, 1
        return clip(cos, -1.0, 1.0) / 2 + 0.5

    def lane_distance_to(self, vehicle, lane: AbstractLane = None) -> float:
        assert self.navigation is not None, "a routing and localization module should be added " \
                                            "to interact with other vehicles"
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    """-------------------------------------- for vehicle making ------------------------------------------"""

    @property
    def LENGTH(self):
        raise NotImplementedError()

    @property
    def HEIGHT(self):
        raise NotImplementedError()

    @property
    def WIDTH(self):
        raise NotImplementedError()

    @property
    def ACTOR_PATH(self):
        raise NotImplementedError()

    @property
    def MOTION_PATH(self):
        raise NotImplementedError()

    def _create_pedestrian_character(self):
        bullet_shape = BulletCylinderShape(self.RADIUS, self.HEIGHT)

        character = BulletCharacterControllerNode(bullet_shape, self.HEIGHT)
        # self.characterNP = get_engine().worldNP.attach_new_node(character)
        # self.characterNP.set_collide_mask(BitMask32.all_on())

        self.dynamic_nodes.append(character)
        self._node_path_list.append(character)

        physics_world = get_engine().physics_world.dynamic_world
        physics_world.attachCharacter(character)

        return character

    def _create_vehicle_chassis(self):
        # self.LENGTH = type(self).LENGTH
        # self.WIDTH = type(self).WIDTH
        # self.HEIGHT = type(self).HEIGHT

        # assert self.LENGTH < BaseVehicle.MAX_LENGTH, "Vehicle is too large!"
        # assert self.WIDTH < BaseVehicle.MAX_WIDTH, "Vehicle is too large!"
        from metaurban.engine.physics_node import BaseRigidBodyNode
        chassis = BaseRigidBodyNode(self.name, MetaUrbanType.VEHICLE)
        self._node_path_list.append(chassis)

        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(vehicle_chassis)
        return vehicle_chassis

    def set_anim_by_rorations(self, rotations):
        # docs: https://docs.panda3d.org/1.10/python/programming/models-and-actors/multi-part-actors
        # GLTF: metaurban/third_party/kitsunetsuki/gltf_inspect.py

        # joints = {"left_shoulder": self.actor.controlJoint(None, 'modelRoot', 'right_shoulder')}
        # self._body.joints['left_shoulder'].setR(clamp((s - 100) / 100) * 120) # in degree

        for rotation in rotations:
            pass

    def _add_visualization(self):
        if self.render:
            [path, scale, offset, HPR] = self.path
            if path not in BaseDeliveryRobot.model_collection:
                car_model = self.loader.loadModel(AssetLoader.file_path("models", path))
                car_model.setTwoSided(False)
                BaseDeliveryRobot.model_collection[path] = car_model
                scale = 0.5
                car_model.setScale(scale)
                # model default, face to y
                HPR = (-90, 0, 0)
                car_model.setHpr(*HPR)
                car_model.setPos(offset[0], offset[1], offset[-1])
                car_model.setZ(offset[-1])
            else:
                car_model = BaseDeliveryRobot.model_collection[path]
            car_model.instanceTo(self.origin)
            self.car_model = car_model
            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (
                        self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.
                    )
                )
                material.setMetallic(self.MATERIAL_METAL_COEFF)
                material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
                material.setRefractiveIndex(1.5)
                material.setRoughness(self.MATERIAL_ROUGHNESS)
                material.setShininess(self.MATERIAL_SHININESS)
                material.setTwoside(False)
                self.origin.setMaterial(material, True)

    def add_navigation(self):
        if self.navigation is not None or self.config["navigation_module"] is None or self.engine.current_map is None:
            return
        navi = self.config["navigation_module"]
        self.navigation = navi(
            # self.engine,
            show_navi_mark=self.config["show_navi_mark"],
            show_dest_mark=self.config["show_dest_mark"],
            show_line_to_dest=self.config["show_line_to_dest"],
            panda_color=self.panda_color,
            name=self.name,
            vehicle_config=self.config
        )

    def reset_navigation(self):
        """
        Update map information that are used by this vehicle, after reset()
        This function will query the map about the spawn position and destination of current vehicle,
        and update the navigation module by feeding the information of spawn point and destination.

        For the spawn position, if it is not specify in the config["spawn_lane_index"], we will automatically
        select one lane based on the localization results.
        """
        if self.navigation is not None and self.config["navigation_module"]:
            self.navigation.reset(self)
            self.navigation.update_localization(self)

    def _state_check(self):
        """
        Check States and filter to update info
        """
        result_1 = self.engine.physics_world.static_world.contactTest(self.chassis.node(), True)
        result_2 = self.engine.physics_world.dynamic_world.contactTest(self.chassis.node(), True)
        contacts = set()
        for contact in result_1.getContacts() + result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            node = node0 if node1.getName() == MetaUrbanType.VEHICLE else node1
            name = node.getName()
            if name == MetaUrbanType.LINE_SOLID_SINGLE_WHITE:
                self.on_white_continuous_line = True
            elif name == MetaUrbanType.LINE_SOLID_SINGLE_YELLOW:
                self.on_yellow_continuous_line = True
            elif name == MetaUrbanType.CROSSWALK:
                self.on_crosswalk = True
            elif name == MetaUrbanType.LINE_BROKEN_SINGLE_YELLOW or name == MetaUrbanType.LINE_BROKEN_SINGLE_WHITE:
                self.on_broken_line = True
            elif name == MetaUrbanType.TRAFFIC_LIGHT:
                light = get_object_from_node(node)
                if light.status == MetaUrbanType.LIGHT_GREEN:
                    self.green_light = True
                elif light.status == MetaUrbanType.LIGHT_RED:
                    self.red_light = True
                elif light.status == MetaUrbanType.LIGHT_YELLOW:
                    self.yellow_light = True
                elif light.status == MetaUrbanType.LIGHT_UNKNOWN:
                    # unknown didn't add
                    continue
                else:
                    raise ValueError("Unknown light status: {}".format(light.status))
                name = light.status
            # they work with the function in collision_callback.py to double-check the collision
            elif name == MetaUrbanType.VEHICLE:
                self.crash_vehicle = True
            elif name == MetaUrbanType.BUILDING:
                self.crash_building = True
            elif MetaUrbanType.is_traffic_object(name):
                self.crash_object = True
            elif name in [MetaUrbanType.PEDESTRIAN, MetaUrbanType.CYCLIST]:
                self.crash_human = True
            else:
                # didn't add
                continue
            contacts.add(name)
        # side walk detect
        res = rect_region_detection(
            self.engine,
            self.position,
            np.rad2deg(self.heading_theta),
            self.LENGTH,
            self.WIDTH,
            CollisionGroup.Sidewalk,
            in_static_world=False  # Sidewalk will be hosted in the dynamic_world. So here we set to False.
        )
        if res.hasHit() and res.getNode().getName() == MetaUrbanType.BOUNDARY_LINE:
            self.crash_sidewalk = True
            contacts.add(MetaUrbanType.BOUNDARY_LINE)

        elif res.hasHit() and res.getNode().getName() == MetaUrbanType.BOUNDARY_SIDEWALK:
            self.crash_sidewalk = True
            contacts.add(MetaUrbanType.BOUNDARY_SIDEWALK)

        elif res.hasHit() and res.getNode().getName() == MetaUrbanType.GUARDRAIL:
            self.crash_sidewalk = True
            contacts.add(MetaUrbanType.GUARDRAIL)

        # elif res.hasHit():
        #     print("Unclassified collision: ", res.getNode().getName())

        # only for visualization detection
        if self.render:
            debug_static_world = self.engine.global_config["debug_static_world"] and self.engine.global_config["debug"]
            res = rect_region_detection(
                self.engine,
                self.position,
                np.rad2deg(self.heading_theta),
                self.LENGTH,
                self.WIDTH,
                CollisionGroup.LaneSurface,
                in_static_world=not debug_static_world
            )
            if not res.hasHit():
                contacts.add(MetaUrbanType.GROUND)
            else:
                if MetaUrbanType.is_lane(res.getNode().getName()):
                    contacts.add(res.getNode().getName())
                else:
                    contacts.add(MetaUrbanType.GROUND)

        self.contact_results.update(contacts)

    def destroy(self):
        super(BaseDeliveryRobot, self).destroy()
        if self.navigation is not None:
            self.navigation.destroy()
        self.navigation = None
        self.wheels = None
        # if self.light is not None:
        #     self.remove_light()

    def set_velocity(self, direction, *args, **kwargs):
        super(BaseDeliveryRobot, self).set_velocity(direction, *args, **kwargs)
        self.last_velocity = self.velocity
        self.last_speed = self.speed

    def set_state(self, state):
        super(BaseDeliveryRobot, self).set_state(state)
        self.set_throttle_brake(float(state["throttle_brake"]))
        self.set_steering(float(state["steering"]))
        self.last_velocity = self.velocity
        self.last_speed = self.speed
        self.last_position = self.position
        self.last_heading_dir = self.heading

    def set_panda_pos(self, pos):
        super(BaseDeliveryRobot, self).set_panda_pos(pos)
        self.last_position = self.position

    def set_position(self, position, height=None):
        super(BaseDeliveryRobot, self).set_position(position, height)
        self.last_position = self.position

    def get_state(self):
        """
        Fetch more information
        """
        state = super(BaseDeliveryRobot, self).get_state()
        state.update(
            {
                "steering": self.steering,
                "throttle_brake": self.throttle_brake,
                "crash_vehicle": self.crash_vehicle,
                "crash_object": self.crash_object,
                "crash_building": self.crash_building,
                "crash_sidewalk": self.crash_sidewalk,
                "size": (self.LENGTH, self.WIDTH, self.HEIGHT),
                "length": self.LENGTH,
                "width": self.WIDTH,
                "height": self.HEIGHT,
            }
        )
        if self.navigation is not None:
            state.update(self.navigation.get_state())
        return state

    # def get_raw_state(self):
    #     ret = dict(position=self.position, heading=self.heading, velocity=self.velocity)
    #     return ret

    def get_dynamics_parameters(self):
        # These two can be changed on the fly
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]

        # These two can only be changed in init
        wheel_friction = self.config["wheel_friction"]
        assert self.max_steering == self.config["max_steering"]
        max_steering = self.max_steering

        mass = self.config["mass"] if self.config["mass"] else self.MASS

        ret = dict(
            max_engine_force=max_engine_force,
            max_brake_force=max_brake_force,
            wheel_friction=wheel_friction,
            max_steering=max_steering,
            mass=mass
        )
        return ret

    def _update_overtake_stat(self):
        lidar_available = self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0
        if self.config["overtake_stat"] and lidar_available:
            surrounding_vs = self.lidar.get_surrounding_vehicles()
            routing = self.navigation
            ckpt_idx = routing._target_checkpoints_index
            for surrounding_v in surrounding_vs:
                if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]
                                                                                                           ]):
                    if self.lane.local_coordinates(self.position)[0] - \
                            self.lane.local_coordinates(surrounding_v.position)[0] < 0:
                        self.front_vehicles.add(surrounding_v)
                        if surrounding_v in self.back_vehicles:
                            self.back_vehicles.remove(surrounding_v)
                    else:
                        self.back_vehicles.add(surrounding_v)
        return {"overtake_vehicle_num": self.get_overtake_num()}

    def get_overtake_num(self):
        return len(self.front_vehicles.intersection(self.back_vehicles))

    def __del__(self):
        super(BaseDeliveryRobot, self).__del__()
        # self.engine = None
        self.navigation = None
        self.wheels = None

    @property
    def reference_lanes(self):
        return self.navigation.current_ref_lanes

    def set_wheel_friction(self, new_friction):
        raise ValueError()
        # for wheel in self.wheels:
        #     wheel.setFrictionSlip(new_friction)

    @property
    def overspeed(self):
        return True if self.lane.speed_limit < self.speed_km_h else False

    @property
    def replay_done(self):
        return self._replay_done if hasattr(self, "_replay_done") else (
            self.crash_building or self.crash_vehicle or
            # self.on_white_continuous_line or
            self.on_yellow_continuous_line
        )

    @property
    def current_action(self):
        return self.last_current_action[-1]

    @property
    def last_action(self):
        return self.last_current_action[0]

    def detach_from_world(self, physics_world):
        if self.navigation is not None:
            self.navigation.detach_from_world()
        super(BaseDeliveryRobot, self).detach_from_world(physics_world)

    def attach_to_world(self, parent_node_path, physics_world):
        if self.config["show_navi_mark"] and self.config["navigation_module"] and self.navigation is not None:
            self.navigation.attach_to_world(self.engine)
        super(BaseDeliveryRobot, self).attach_to_world(parent_node_path, physics_world)

    def set_break_down(self, break_down=True):
        self.break_down = break_down
        # self.set_static(True)

    @property
    def max_speed_km_h(self):
        return self.config["max_speed_km_h"]

    @property
    def max_speed_m_s(self):
        return self.config["max_speed_km_h"] / 3.6

    @property
    def top_down_length(self):
        return self.config["top_down_length"] if self.config["top_down_length"] else self.LENGTH

    @property
    def top_down_width(self):
        return self.config["top_down_width"] if self.config["top_down_width"] else self.WIDTH

    @property
    def lane(self):
        return self.navigation.current_lane

    @property
    def lane_index(self):
        return self.navigation.current_lane.index

    @property
    def panda_color(self):
        c = super(BaseDeliveryRobot, self).panda_color
        # if self._use_special_color:
        #     color = sns.color_palette("colorblind")
        #     rand_c = color[2]  # A pretty green
        #     c = rand_c
        return c

    def before_reset(self):
        for obj in [self.navigation]:
            if obj is not None and hasattr(obj, "before_reset"):
                obj.before_reset()

    """------------------------------------------- overwrite -------------------------------------------------"""

    def convert_to_world_coordinates(self, vector, origin):
        return super(BaseDeliveryRobot, self).convert_to_world_coordinates([-vector[-1], vector[0]], origin)

    def convert_to_local_coordinates(self, vector, origin):
        ret = super(BaseDeliveryRobot, self).convert_to_local_coordinates(vector, origin)
        return np.array([ret[1], -ret[0]])

    @property
    def heading_theta(self):
        return wrap_to_pi(super(BaseDeliveryRobot, self).heading_theta + np.pi / 2)

    def set_heading_theta(self, heading_theta, in_rad=True) -> None:
        """
        Set heading theta for this object. Vehicle local frame has a 90 degree offset
        :param heading_theta: float in rad
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """
        super(BaseDeliveryRobot, self).set_heading_theta(heading_theta - np.pi / 2, in_rad)
        self.last_heading_dir = self.heading

    @property
    def roll(self):
        """
        Return the roll of this object
        """
        return np.deg2rad(self.origin.getR())

    def set_roll(self, roll):
        self.origin.setR(roll)

    @property
    def pitch(self):
        """
        Return the pitch of this object
        """
        return np.deg2rad(self.origin.getP())

    def set_pitch(self, pitch):
        self.origin.setP(pitch)

    def show_coordinates(self):
        if self.coordinates_debug_np is not None:
            self.coordinates_debug_np.reparentTo(self.origin)
            return
        height = self.HEIGHT + 0.2
        self.coordinates_debug_np = NodePath("debug coordinate")
        self.coordinates_debug_np.hide(CamMask.AllOn)
        self.coordinates_debug_np.show(CamMask.MainCam)
        # 90 degrees offset
        x = self.engine._draw_line_3d([0, 0, height], [0, 2, height], [1, 1, 1, 1], 3)
        y = self.engine._draw_line_3d([0, 0, height], [-1, 0, height], [1, 1, 1, 1], 3)
        z = self.engine._draw_line_3d([0, 0, height], [0, 0, height + 0.5], [1, 1, 1, 1], 3)
        x.reparentTo(self.coordinates_debug_np)
        y.reparentTo(self.coordinates_debug_np)
        z.reparentTo(self.coordinates_debug_np)
        self.coordinates_debug_np.reparentTo(self.origin)

    @property
    def lidar(self):
        return self.engine.get_sensor("lidar")

    @property
    def side_detector(self):
        return self.engine.get_sensor("side_detector")

    @property
    def lane_line_detector(self):
        return self.engine.get_sensor("lane_line_detector")


class EgoDeliveryRobot(BaseObject, BaseDeliveryRobotState):
    """
    Vehicle chassis and its wheels index
                    0       1
                    II-----II
                        |
                        |  <---chassis/wheelbase
                        |
                    II-----II
                    2       3
    """
    COLLISION_MASK = CollisionGroup.Vehicle
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
    MAX_LENGTH = 10
    MAX_WIDTH = 2.5
    MAX_STEERING = 80
    SEMANTIC_LABEL = Semantics.CAR.label

    # LENGTH = None
    # WIDTH = None
    # HEIGHT = None

    TIRE_RADIUS = None
    TIRE_MODEL_CORRECT = 1  # correct model left right error
    TIRE_TWO_SIDED = False  # tires for some vehicles need two-sided enabled for correctly visualization
    LATERAL_TIRE_TO_CENTER = None
    TIRE_WIDTH = 0.2
    FRONT_WHEELBASE = None
    REAR_WHEELBASE = None
    LIGHT_POSITION = (-0.67, 1.62, 0.05)

    # MASS = None

    CHASSIS_TO_WHEEL_AXIS = 0.1
    SUSPENSION_LENGTH = 15
    SUSPENSION_STIFFNESS = 40

    # for random color choosing
    MATERIAL_COLOR_COEFF = 1.6  # to resist other factors, since other setting may make color dark
    MATERIAL_METAL_COEFF = 0.1  # 0-1
    MATERIAL_ROUGHNESS = 0.8  # smaller to make it more smooth, and reflect more light
    MATERIAL_SHININESS = 128  # 0-128 smaller to make it more smooth, and reflect more light
    MATERIAL_SPECULAR_COLOR = (3, 3, 3, 3)

    # control
    STEERING_INCREMENT = 0.05

    # save memory, load model once
    model_collection = {}
    path = None

    def __init__(
        self,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
        position=None,
        heading=None,
        _calling_reset=True,
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param vehicle_config: mostly, vehicle module config
        :param random_seed: int
        """
        # check
        assert vehicle_config is not None, "Please specify the vehicle config."
        assert engine_initialized(), "Please make sure game engine is successfully initialized!"

        # NOTE: it is the game engine, not vehicle drivetrain
        # self.engine = get_engine()
        BaseObject.__init__(self, name, random_seed, self.engine.global_config["vehicle_config"])
        BaseDeliveryRobotState.__init__(self)
        self.update_config(vehicle_config)
        self.set_metaurban_type(MetaUrbanType.VEHICLE)
        use_special_color = self.config["use_special_color"]

        # build vehicle physics model
        vehicle_chassis = self._create_vehicle_chassis()
        self.add_body(vehicle_chassis.getChassis())
        self.system = vehicle_chassis
        self.chassis = self.origin
        self.wheels = self._create_wheel()

        # light experimental!
        self.light = None
        self._light_direction_queue = None
        self._light_models = None
        self.light_name = None

        # powertrain config
        self.enable_reverse = self.config["enable_reverse"]
        self.policy_reverse = self.config.get('policy_reverse', False)
        self.max_steering = self.config["max_steering"]

        # visualization
        self._use_special_color = use_special_color
        self._add_visualization()

        # navigation module
        self.navigation: Optional[NodeNetworkNavigation] = None

        # state info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = (0, 0)
        self.last_heading_dir = self.heading
        self.dist_to_left_side = None
        self.dist_to_right_side = None
        self.last_velocity = 0
        self.last_speed = 0

        # step info
        self.out_of_route = None
        self.on_lane = None
        self.spawn_place = (0, 0)
        self._init_step_info()

        # others
        self.takeover = False
        self.expert_takeover = False
        self.energy_consumption = 0
        self.break_down = False

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()

        # if self.engine.current_map is not None:
        if _calling_reset:
            self.reset(position=position, heading=heading, vehicle_config=vehicle_config)

    def _init_step_info(self):
        # done info will be initialized every frame
        self.init_state_info()
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not

    @staticmethod
    def _preprocess_action(action):
        if action is None:
            return None, {"raw_action": None}
        action = safe_clip_for_small_array(action, -1, 1)
        return action, {'raw_action': (action[0], action[1])}

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        # if action is None:
        #     action = [0, 0]

        self._init_step_info()
        action, step_info = self._preprocess_action(action)

        self.last_position = self.position  # 2D vector
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar
        self.last_heading_dir = self.heading
        if action is not None:
            self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        # if self.increment_steering:
        #     self._set_incremental_action(action)
        # else:
        self._set_action(action)
        return step_info

    def after_step(self):
        if self.navigation and self.config["ego_navigation_module"]:
            # pass
            self.navigation.update_localization(self)
        # self._state_check()
        # self.update_dist_to_left_right()
        # step_energy, episode_energy = self._update_energy_consumption()
        # self.out_of_route = self._out_of_route()
        # step_info = self._update_overtake_stat()
        # my_policy = self.engine.get_policy(self.name)
        # step_info.update(
        #     {
        #         "velocity": float(self.speed),
        #         "steering": float(self.steering),
        #         "acceleration": float(self.throttle_brake),
        #         "step_energy": step_energy,
        #         "episode_energy": episode_energy,
        #         "policy": my_policy.name if my_policy is not None else my_policy
        #     }
        # )
        # return step_info

    def _out_of_route(self):
        left, right = self._dist_to_route_left_right()
        return True if right < 0 or left < 0 else False

    def _update_energy_consumption(self):
        """
        The calculation method is from
        https://www.researchgate.net/publication/262182035_Reduction_of_Fuel_Consumption_and_Exhaust_Pollutant_Using_Intelligent_Transport_System
        default: 3rd gear, try to use ae^bx to fit it, dp: (90, 8), (130, 12)
        :return: None
        """
        distance = norm(self.last_position[0] - self.position[0], self.last_position[1] - self.position[1]) / 1000  # km
        step_energy = 3.25 * math.pow(np.e, 0.01 * self.speed_km_h) * distance / 100
        # step_energy is in Liter, we return mL
        step_energy = step_energy * 1000
        self.energy_consumption += step_energy  # L/100 km
        return step_energy, self.energy_consumption

    def reset(
        self,
        vehicle_config=None,
        name=None,
        random_seed=None,
        position: np.ndarray = None,
        heading: float = 0.0,
        *args,
        **kwargs
    ):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if name is not None:
            self.rename(name)

        # reset fully
        self.update_config(self.engine.global_config["vehicle_config"])
        if random_seed is not None:
            assert isinstance(random_seed, int)
            self.seed(random_seed)
            self.sample_parameters()

        if vehicle_config is not None:
            self.update_config(vehicle_config)
        from metaurban.component.delivery_robot.deliveryrobot_type import DeliveryRobot_class_to_type
        self.config["vehicle_model"] = DeliveryRobot_class_to_type[self.__class__]

        # Update some modules that might not be initialized before
        self.add_navigation()
        self.reset_navigation()

        self.set_pitch(0)
        self.set_roll(0)
        position = self.navigation.init_position

        # position = [0, 0]
        self.navigation.reference_trajectory
        heading = np.arctan2(
            self.navigation.position_list[1][1] - self.navigation.position_list[0][1],
            self.navigation.position_list[1][0] - self.navigation.position_list[0][0]
        )
        # heading = 0.
        if self.engine.global_config['test_terrain_system'] or self.engine.global_config[
                'test_slope_system'] or self.engine.global_config['test_rough_system']:
            position = [10, 0]
            heading = np.pi / 2.
        self.spawn_place = position
        # print("position:", position)
        self.set_heading_theta(heading)
        self.set_static(False)
        # self.set_wheel_friction(self.config["wheel_friction"])

        if len(position) == 2:
            self.set_position(position, height=self.HEIGHT / 2)
        elif len(position) == 3:
            self.set_position(position[:2], height=position[-1])
        else:
            raise ValueError()

        self.body.clearForces()
        self.body.setLinearVelocity(Vec3(0, 0, 0))
        self.body.setAngularVelocity(Vec3(0, 0, 0))
        self.system.resetSuspension()
        self._apply_throttle_brake(0.0)
        # np.testing.assert_almost_equal(self.position, pos, decimal=4)

        # done info
        self._init_step_info()

        # other info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = self.spawn_place
        self.last_heading_dir = self.heading
        self.last_velocity = self.velocity  # 2D vector
        self.last_speed = self.speed  # Scalar

        self.update_dist_to_left_right()
        self.takeover = False
        self.energy_consumption = 0

        # overtake_stat
        self.front_vehicles = set()
        self.back_vehicles = set()
        self.expert_takeover = False
        if self.config["ego_navigation_module"] and self.engine.current_map is not None:
            assert self.navigation

        if self.config["spawn_velocity"] is not None:
            self.set_velocity(self.config["spawn_velocity"], in_local_frame=self.config["spawn_velocity_car_frame"])

        # clean lights
        if self.config["light"]:
            self.add_light()
        else:
            self.remove_light()

        # self.add_light()

    """------------------------------------------- act -------------------------------------------------"""

    def remove_light(self):
        if self.light is not None:
            if self.use_render_pipeline:
                self.engine.render_pipeline.remove_light(self.light)
                self.engine.taskMgr.remove(self.light_name)
            self.light_name = None
            self.light = None
            for m in self._light_models:
                m.removeNode()
            self._light_models = None
            self._light_direction_queue = None

    def add_light(self):
        """
        Experimental feature
        """
        # assert self.use_render_pipeline, "Can be Enabled when using render pipeline"
        if self.light is None:
            self._light_models = []
            for y in [-1, 1]:
                light_model = self.loader.loadModel(AssetLoader.file_path("models", "sphere.egg"))
                light_model.reparentTo(self.origin)
                light_model.setPos(self.LIGHT_POSITION[0] * y, self.LIGHT_POSITION[1], self.LIGHT_POSITION[2])
                light_model.setScale(0.13, 0.05, 0.13)
                material = Material()
                material.setBaseColor((1, 1, 1, 1))
                material.setShininess(128)
                material.setEmission((1, 1, 1, 1))
                light_model.setMaterial(material, True)
                self._light_models.append(light_model)
            if self.use_render_pipeline:
                self.light_name = "light_{}".format(self.id)
                self.light = RPSpotLight()
                self.light.setColorFromTemperature(3 * 1000.0)
                self.light.setRadius(500)
                self.light.setFov(100)
                self.light.energy = 600
                self.light.casts_shadows = False
                self.light.shadow_map_resolution = 128
                self.engine.render_pipeline.add_light(self.light)
                self.engine.taskMgr.add(self._update_light_pos, self.light_name)
                self._light_direction_queue = []

    def _update_light_pos(self, task):
        pos = self.convert_to_world_coordinates([self.LENGTH / 2, 0], self.position)
        self.light.set_pos(*pos, self.get_z())
        self._light_direction_queue.append([*self.heading, 0])
        idx = min(len(self._light_direction_queue), 50)
        pos = np.mean(self._light_direction_queue[-idx:], axis=0)
        self.light.set_direction(pos[0], pos[1], pos[2])
        return task.cont

    def set_steering(self, steering):
        steering = float(steering)
        self.system.setSteeringValue(steering, 0)
        self.system.setSteeringValue(steering, 1)
        self.steering = steering

    def set_throttle_brake(self, throttle_brake):
        throttle_brake = float(throttle_brake)
        self._apply_throttle_brake(throttle_brake)
        self.throttle_brake = throttle_brake

    def _set_action(self, action):
        if action is None:
            return
        steering = action[0]
        self.throttle_brake = action[1]
        self.steering = steering
        self.system.setSteeringValue(self.steering * self.max_steering, 0)
        self.system.setSteeringValue(self.steering * self.max_steering, 1)
        self._apply_throttle_brake(action[1])

    def _set_incremental_action(self, action: np.ndarray):
        if action is None:
            return
        self.throttle_brake = action[1]
        self.steering += action[0] * self.STEERING_INCREMENT
        self.steering = clip(self.steering, -1, 1)
        steering = self.steering * self.max_steering
        self.system.setSteeringValue(steering, 0)
        self.system.setSteeringValue(steering, 1)
        self._apply_throttle_brake(action[1])

    def _apply_throttle_brake(self, throttle_brake):
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]
        for wheel_index in range(4):
            if throttle_brake >= 0:
                self.system.setBrake(2.0, wheel_index)
                if self.speed_km_h > self.max_speed_km_h:
                    self.system.applyEngineForce(0.0, wheel_index)
                else:
                    self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
            else:
                if self.engine.global_config['manual_control']:
                    if self.engine.current_track_agent.expert_takeover:
                        if self.policy_reverse:
                            self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
                            self.system.setBrake(0, wheel_index)
                        else:
                            DEADZONE = 0.01

                            # Speed m/s in car's heading:
                            heading = self.heading
                            velocity = self.velocity
                            speed_in_heading = velocity[0] * heading[0] + velocity[1] * heading[1]

                            if speed_in_heading < DEADZONE:
                                self.system.applyEngineForce(0.0, wheel_index)
                                self.system.setBrake(2, wheel_index)
                            else:
                                self.system.applyEngineForce(0.0, wheel_index)
                                self.system.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)
                    else:
                        if self.enable_reverse:
                            self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
                            self.system.setBrake(0, wheel_index)
                        else:
                            DEADZONE = 0.01

                            # Speed m/s in car's heading:
                            heading = self.heading
                            velocity = self.velocity
                            speed_in_heading = velocity[0] * heading[0] + velocity[1] * heading[1]

                            if speed_in_heading < DEADZONE:
                                self.system.applyEngineForce(0.0, wheel_index)
                                self.system.setBrake(2, wheel_index)
                            else:
                                self.system.applyEngineForce(0.0, wheel_index)
                                self.system.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)
                else:
                    if self.enable_reverse:
                        self.system.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
                        self.system.setBrake(0, wheel_index)
                    else:
                        DEADZONE = 0.01

                        # Speed m/s in car's heading:
                        heading = self.heading
                        velocity = self.velocity
                        speed_in_heading = velocity[0] * heading[0] + velocity[1] * heading[1]

                        if speed_in_heading < DEADZONE:
                            self.system.applyEngineForce(0.0, wheel_index)
                            self.system.setBrake(2, wheel_index)
                        else:
                            self.system.applyEngineForce(0.0, wheel_index)
                            self.system.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def update_dist_to_left_right(self):
        self.dist_to_left_side, self.dist_to_right_side = self._dist_to_route_left_right()

    def _dist_to_route_left_right(self):
        # TODO
        if self.navigation is None or self.navigation.current_ref_lanes is None:
            return 0, 0
        current_reference_lane = self.navigation.current_ref_lanes[0]
        _, lateral_to_reference = current_reference_lane.local_coordinates(self.position)
        lateral_to_left = lateral_to_reference + self.navigation.get_current_lane_width() / 2
        lateral_to_right = self.navigation.get_current_lateral_range(self.position, self.engine) - lateral_to_left
        return lateral_to_left, lateral_to_right

    # @property
    # def heading_theta(self):
    #     """
    #     Get the heading theta of vehicle, unit [rad]
    #     :return:  heading in rad
    #     """
    #     return wrap_to_pi(self.origin.getH() / 180 * math.pi)

    # @property
    # def velocity(self) -> np.ndarray:
    #     return self.speed * self.velocity_direction
    #
    # @property
    # def velocity_km_h(self) -> np.ndarray:
    #     return self.speed * self.velocity_direction * 3.6

    @property
    def chassis_velocity_direction(self):
        raise DeprecationWarning(
            "This API returns the direction of velocity which is approximately heading direction. "
            "Deprecate it and make things easy"
        )
        # direction = self.system.getForwardVector()
        # return np.asarray([direction[0], direction[1]])

    """---------------------------------------- some math tool ----------------------------------------------"""

    def heading_diff(self, target_lane):
        lateral = None
        if isinstance(target_lane, StraightLane):
            lateral = np.asarray(get_vertical_vector(target_lane.end - target_lane.start)[1])
        elif isinstance(target_lane, CircularLane):
            if not target_lane.is_clockwise():
                lateral = self.position - target_lane.center
            else:
                lateral = target_lane.center - self.position
        elif isinstance(target_lane, PointLane):
            lateral = target_lane.lateral_direction(target_lane.local_coordinates(self.position)[0])

        lateral_norm = norm(lateral[0], lateral[1])
        forward_direction = self.heading
        # print(f"Old forward direction: {self.forward_direction}, new heading {self.heading}")
        forward_direction_norm = norm(forward_direction[0], forward_direction[1])
        if not lateral_norm * forward_direction_norm:
            return 0
        cos = (
            (forward_direction[0] * lateral[0] + forward_direction[1] * lateral[1]) /
            (lateral_norm * forward_direction_norm)
        )
        # return cos
        # Normalize to 0, 1
        return clip(cos, -1.0, 1.0) / 2 + 0.5

    def lane_distance_to(self, vehicle, lane: AbstractLane = None) -> float:
        assert self.navigation is not None, "a routing and localization module should be added " \
                                            "to interact with other vehicles"
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    """-------------------------------------- for vehicle making ------------------------------------------"""

    @property
    def LENGTH(self):
        raise NotImplementedError()

    @property
    def HEIGHT(self):
        raise NotImplementedError()

    @property
    def WIDTH(self):
        raise NotImplementedError()

    def _create_vehicle_chassis(self):
        # self.LENGTH = type(self).LENGTH
        # self.WIDTH = type(self).WIDTH
        # self.HEIGHT = type(self).HEIGHT

        # assert self.LENGTH < EgoDeliveryRobot.MAX_LENGTH, "Vehicle is too large!"
        # assert self.WIDTH < EgoDeliveryRobot.MAX_WIDTH, "Vehicle is too large!"
        from metaurban.engine.physics_node import BaseRigidBodyNode
        chassis = BaseRigidBodyNode(self.name, MetaUrbanType.VEHICLE)
        self._node_path_list.append(chassis)

        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(vehicle_chassis)
        return vehicle_chassis

    def _add_visualization(self):
        if self.render:
            [path, scale, offset, HPR] = self.path
            if path not in EgoDeliveryRobot.model_collection:
                car_model = self.loader.loadModel(AssetLoader.file_path("models", path))
                car_model.setTwoSided(False)
                EgoDeliveryRobot.model_collection[path] = car_model
                # scale = 0.15 # for white one
                scale = 1.5  # for gradient
                car_model.setScale(scale)
                # model default, face to y
                HPR = (180, 0, 0)
                car_model.setHpr(*HPR)
                if 'coco' in path:
                    car_model.setPos(offset[0] - 3.3, offset[1], offset[-1])
                elif 'free3DVersion' in path:
                    car_model.setPos(offset[0], offset[1], offset[-1])
                car_model.setZ(-self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS + offset[-1])
                from panda3d.core import TextNode
                from panda3d.core import TransparencyAttrib

            else:
                car_model = EgoDeliveryRobot.model_collection[path]

            # text_node = TextNode('3d_text')
            # text_node.setText("Partially Transparent")
            # text_node.setTextColor(1, 0, 0, 1)
            # text_node.setAlign(TextNode.ACenter)  # 文字居中对齐

            # 设置文字挤出，以便成为3D效果
            # text_node.setExtrudeDepth(0.1)  # 文字的厚度
            # text_node.setCardColor(0.1, 0.1, 0.1, 0.1)
            # text_node.setCardAsMargin(0.5, 0.5, 0.5, 0.5)
            # text_node.setCardDecal(True)
            # # text_node.setFrameColor(1, 1, 1, 1)  # 文字框的颜色
            # # text_node.setFrameAsMargin(0.5, 0.5, 0.5, 0.5)  # 设置文字的边框

            # # 创建 NodePath
            # text_np = NodePath(text_node)
            # text_np.reparentTo(car_model)  # 将文字附加到场景图

            # # 调整 3D 文字的位置和缩放
            # text_np.setScale(0.5)
            # text_np.setPos(0, 5, 1)
            # text_np.setHpr(*(0, 0, 0))
            # text_np.setDepthTest(False)
            # text_np.setDepthWrite(False)
            # self._node_path_list.append(text_np)
            # text_np.setTransparency(TransparencyAttrib.MAlpha)

            car_model.instanceTo(self.origin)
            self.car_model = car_model
            # text_np.setTransparency(TransparencyAttrib.MAlpha)
            # self.text_node = TextNode('target_text')
            # self.text_node.setText("Target")
            # self.text_node.setTextColor(1, 1, 0, 1)  # 设置为黄色以更明显
            # self.text_node.setCardColor(0.1, 0.1, 0.1, 0.5)  # 设置背景颜色
            # self.text_node.setCardAsMargin(0.2, 0.2, 0.2, 0.2)  # 设置文本的边框
            # self.text_node.setCardDecal(True)
            # textNodePath = self.origin.attachNewNode(self.text_node )
            # self._node_path_list.append(textNodePath)
            # textNodePath.setScale(0.052)
            # text = self.text_node
            # text.setFrameColor(0, 0, 0, 1)
            # text.setTextColor(0, 0, 0, 1)
            # text.setFrameAsMargin(0.5, 0.5, 0.5, 0.5)
            # text.setAlign(TextNode.ARight)
            # textNodePath.setPos(-1.125111, 0, 0.9)

            # textNodePath.setScale(0.052)
            # text.setFrameColor(0, 0, 0, 1)
            # text.setTextColor(0, 0, 0, 1)
            # text.setFrameAsMargin(-self.GAP, self.PARA_VIS_LENGTH, 0, 0)
            # text.setAlign(TextNode.ARight)
            # textNodePath.setPos(-1.125111, 0, 0.9 - i * 0.08)

            # text_np = NodePath(self.text_node)
            # text_np.reparentTo(car_model)  # 将文本附加到模型
            # self._node_path_list.append(text_np)
            # text_np.setScale(0.5)  # 缩放到合适大小，可能需要继续调整
            # text_np.setPos(0, 0, 15)  # 根据模型大小调整位置

            # # 启用透明度处理
            # text_np.setTransparency(TransparencyAttrib.MAlpha)

            # # 设置渲染顺序，确保文本在模型前面渲染
            # text_np.setBin('fixed', 0)
            # text_np.setDepthTest(False)
            # text_np.setDepthWrite(False)
            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (
                        self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.
                    )
                )
                material.setMetallic(self.MATERIAL_METAL_COEFF)
                material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
                material.setRefractiveIndex(1.5)
                material.setRoughness(self.MATERIAL_ROUGHNESS)
                material.setShininess(self.MATERIAL_SHININESS)
                material.setTwoside(False)
                self.origin.setMaterial(material, True)

    def _create_wheel(self):
        f_l = self.FRONT_WHEELBASE
        r_l = -self.REAR_WHEELBASE
        lateral = self.LATERAL_TIRE_TO_CENTER
        axis_height = self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS
        radius = self.TIRE_RADIUS
        wheels = []
        for k, pos in enumerate([Vec3(lateral, f_l, axis_height), Vec3(-lateral, f_l, axis_height),
                                 Vec3(lateral, r_l, axis_height), Vec3(-lateral, r_l, axis_height)]):
            wheel = self._add_wheel(pos, radius, True if k < 2 else False, True if k == 0 or k == 2 else False)
            wheels.append(wheel)
        return wheels

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        self._node_path_list.append(wheel_np)

        wheel = self.system.createWheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))

        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(self.SUSPENSION_LENGTH)
        wheel.setSuspensionStiffness(self.SUSPENSION_STIFFNESS)
        wheel.setWheelsDampingRelaxation(4.8)
        wheel.setWheelsDampingCompression(1.2)
        wheel_friction = self.config["wheel_friction"] if not self.config["no_wheel_friction"] else 0
        wheel.setFrictionSlip(wheel_friction)
        wheel.setRollInfluence(0.5)
        return wheel

    def add_navigation(self):
        if self.navigation is not None or self.config["ego_navigation_module"
                                                      ] is None or self.engine.current_map is None:
            return
        navi = self.config["ego_navigation_module"]
        self.navigation = navi(
            # self.engine,
            show_navi_mark=self.config["show_navi_mark"],
            show_dest_mark=self.config["show_dest_mark"],
            show_line_to_dest=self.config["show_line_to_dest"],
            panda_color=self.panda_color,
            name=self.name,
            vehicle_config=self.config
        )

    def reset_navigation(self):
        """
        Update map information that are used by this vehicle, after reset()
        This function will query the map about the spawn position and destination of current vehicle,
        and update the navigation module by feeding the information of spawn point and destination.

        For the spawn position, if it is not specify in the config["spawn_lane_index"], we will automatically
        select one lane based on the localization results.
        """
        if self.navigation is not None and self.config["ego_navigation_module"]:
            self.navigation.reset(self)
            self.navigation.update_localization(self)
        # if self.navigation is not None and self.config["ego_navigation_module"]:
        #     navi = self.config["ego_navigation_module"]
        #     self.navigation = navi(
        #         # self.engine,
        #         show_navi_mark=self.config["show_navi_mark"],
        #         show_dest_mark=self.config["show_dest_mark"],
        #         show_line_to_dest=self.config["show_line_to_dest"],
        #         panda_color=self.panda_color,
        #         name=self.name,
        #         vehicle_config=self.config
        #     )

    def _state_check(self):
        """
        Check States and filter to update info
        """
        result_1 = self.engine.physics_world.static_world.contactTest(self.chassis.node(), True)
        result_2 = self.engine.physics_world.dynamic_world.contactTest(self.chassis.node(), True)
        contacts = set()
        for contact in result_1.getContacts() + result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            node = node0 if node1.getName() == MetaUrbanType.VEHICLE else node1
            name = node.getName()
            if name == MetaUrbanType.LINE_SOLID_SINGLE_WHITE:
                self.on_white_continuous_line = True
            elif name == MetaUrbanType.LINE_SOLID_SINGLE_YELLOW:
                self.on_yellow_continuous_line = True
            elif name == MetaUrbanType.CROSSWALK:
                self.on_crosswalk = True
            elif name == MetaUrbanType.LINE_BROKEN_SINGLE_YELLOW or name == MetaUrbanType.LINE_BROKEN_SINGLE_WHITE:
                self.on_broken_line = True
            elif name == MetaUrbanType.TRAFFIC_LIGHT:
                light = get_object_from_node(node)
                if light.status == MetaUrbanType.LIGHT_GREEN:
                    self.green_light = True
                elif light.status == MetaUrbanType.LIGHT_RED:
                    self.red_light = True
                elif light.status == MetaUrbanType.LIGHT_YELLOW:
                    self.yellow_light = True
                elif light.status == MetaUrbanType.LIGHT_UNKNOWN:
                    # unknown didn't add
                    continue
                else:
                    raise ValueError("Unknown light status: {}".format(light.status))
                name = light.status
            # they work with the function in collision_callback.py to double-check the collision
            elif name == MetaUrbanType.VEHICLE:
                self.crash_vehicle = True
            elif name == MetaUrbanType.BUILDING:
                self.crash_building = True
            elif MetaUrbanType.is_traffic_object(name):
                self.crash_object = True
            elif name in [MetaUrbanType.PEDESTRIAN, MetaUrbanType.CYCLIST]:
                self.crash_human = True
            else:
                # didn't add
                continue
            contacts.add(name)
        # side walk detect
        res = rect_region_detection(
            self.engine,
            self.position,
            np.rad2deg(self.heading_theta),
            self.LENGTH,
            self.WIDTH,
            CollisionGroup.Sidewalk,
            in_static_world=False  # Sidewalk will be hosted in the dynamic_world. So here we set to False.
        )
        if res.hasHit() and res.getNode().getName() == MetaUrbanType.BOUNDARY_LINE:
            self.crash_sidewalk = True
            contacts.add(MetaUrbanType.BOUNDARY_LINE)

        elif res.hasHit() and res.getNode().getName() == MetaUrbanType.BOUNDARY_SIDEWALK:
            self.crash_sidewalk = True
            contacts.add(MetaUrbanType.BOUNDARY_SIDEWALK)

        elif res.hasHit() and res.getNode().getName() == MetaUrbanType.GUARDRAIL:
            self.crash_sidewalk = True
            contacts.add(MetaUrbanType.GUARDRAIL)

        # elif res.hasHit():
        #     print("Unclassified collision: ", res.getNode().getName())

        # only for visualization detection
        if self.render:
            debug_static_world = self.engine.global_config["debug_static_world"] and self.engine.global_config["debug"]
            res = rect_region_detection(
                self.engine,
                self.position,
                np.rad2deg(self.heading_theta),
                self.LENGTH,
                self.WIDTH,
                CollisionGroup.LaneSurface,
                in_static_world=not debug_static_world
            )
            if not res.hasHit():
                contacts.add(MetaUrbanType.GROUND)
            else:
                if MetaUrbanType.is_lane(res.getNode().getName()):
                    contacts.add(res.getNode().getName())
                else:
                    contacts.add(MetaUrbanType.GROUND)

        self.contact_results.update(contacts)

    def destroy(self):
        super(EgoDeliveryRobot, self).destroy()
        if self.navigation is not None:
            self.navigation.destroy()
        self.navigation = None
        self.wheels = None
        if self.light is not None:
            self.remove_light()

    def set_velocity(self, direction, *args, **kwargs):
        super(EgoDeliveryRobot, self).set_velocity(direction, *args, **kwargs)
        self.last_velocity = self.velocity
        self.last_speed = self.speed

    def set_state(self, state):
        super(EgoDeliveryRobot, self).set_state(state)
        self.set_throttle_brake(float(state["throttle_brake"]))
        self.set_steering(float(state["steering"]))
        self.last_velocity = self.velocity
        self.last_speed = self.speed
        self.last_position = self.position
        self.last_heading_dir = self.heading

    def set_panda_pos(self, pos):
        super(EgoDeliveryRobot, self).set_panda_pos(pos)
        self.last_position = self.position

    def set_position(self, position, height=None):
        super(EgoDeliveryRobot, self).set_position(position, height)
        self.last_position = self.position

    def get_state(self):
        """
        Fetch more information
        """
        state = super(EgoDeliveryRobot, self).get_state()
        state.update(
            {
                "steering": self.steering,
                "throttle_brake": self.throttle_brake,
                "crash_vehicle": self.crash_vehicle,
                "crash_object": self.crash_object,
                "crash_building": self.crash_building,
                "crash_sidewalk": self.crash_sidewalk,
                "size": (self.LENGTH, self.WIDTH, self.HEIGHT),
                "length": self.LENGTH,
                "width": self.WIDTH,
                "height": self.HEIGHT,
            }
        )
        if self.navigation is not None:
            state.update(self.navigation.get_state())
        return state

    # def get_raw_state(self):
    #     ret = dict(position=self.position, heading=self.heading, velocity=self.velocity)
    #     return ret

    def get_dynamics_parameters(self):
        # These two can be changed on the fly
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]

        # These two can only be changed in init
        wheel_friction = self.config["wheel_friction"]
        assert self.max_steering == self.config["max_steering"]
        max_steering = self.max_steering

        mass = self.config["mass"] if self.config["mass"] else self.MASS

        ret = dict(
            max_engine_force=max_engine_force,
            max_brake_force=max_brake_force,
            wheel_friction=wheel_friction,
            max_steering=max_steering,
            mass=mass
        )
        return ret

    def _update_overtake_stat(self):
        lidar_available = self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0
        if self.config["overtake_stat"] and lidar_available:
            surrounding_vs = self.lidar.get_surrounding_vehicles()
            routing = self.navigation
            ckpt_idx = routing._target_checkpoints_index
            for surrounding_v in surrounding_vs:
                if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]
                                                                                                           ]):
                    if self.lane.local_coordinates(self.position)[0] - \
                            self.lane.local_coordinates(surrounding_v.position)[0] < 0:
                        self.front_vehicles.add(surrounding_v)
                        if surrounding_v in self.back_vehicles:
                            self.back_vehicles.remove(surrounding_v)
                    else:
                        self.back_vehicles.add(surrounding_v)
        return {"overtake_vehicle_num": self.get_overtake_num()}

    def get_overtake_num(self):
        return len(self.front_vehicles.intersection(self.back_vehicles))

    def __del__(self):
        super(EgoDeliveryRobot, self).__del__()
        # self.engine = None
        self.navigation = None
        self.wheels = None

    @property
    def reference_lanes(self):
        return self.navigation.current_ref_lanes

    def set_wheel_friction(self, new_friction):
        raise ValueError()
        # for wheel in self.wheels:
        #     wheel.setFrictionSlip(new_friction)

    @property
    def overspeed(self):
        return True if self.lane.speed_limit < self.speed_km_h else False

    @property
    def replay_done(self):
        return self._replay_done if hasattr(self, "_replay_done") else (
            self.crash_building or self.crash_vehicle or
            # self.on_white_continuous_line or
            self.on_yellow_continuous_line
        )

    @property
    def current_action(self):
        return self.last_current_action[-1]

    @property
    def last_action(self):
        return self.last_current_action[0]

    def detach_from_world(self, physics_world):
        if self.navigation is not None:
            self.navigation.detach_from_world()
        super(EgoDeliveryRobot, self).detach_from_world(physics_world)

    def attach_to_world(self, parent_node_path, physics_world):
        if self.config["show_navi_mark"] and self.config["ego_navigation_module"] and self.navigation is not None:
            self.navigation.attach_to_world(self.engine)
        super(EgoDeliveryRobot, self).attach_to_world(parent_node_path, physics_world)

    def set_break_down(self, break_down=True):
        self.break_down = break_down
        # self.set_static(True)

    @property
    def max_speed_km_h(self):
        return self.config["max_speed_km_h"]

    @property
    def max_speed_m_s(self):
        return self.config["max_speed_km_h"] / 3.6

    @property
    def top_down_length(self):
        return self.config["top_down_length"] if self.config["top_down_length"] else self.LENGTH

    @property
    def top_down_width(self):
        return self.config["top_down_width"] if self.config["top_down_width"] else self.WIDTH

    @property
    def lane(self):
        return self.navigation.current_lane

    @property
    def lane_index(self):
        return self.navigation.current_lane.index

    @property
    def panda_color(self):
        c = super(EgoDeliveryRobot, self).panda_color
        if self._use_special_color:
            color = sns.color_palette("colorblind")
            rand_c = color[2]  # A pretty green
            c = rand_c
        return c

    def before_reset(self):
        for obj in [self.navigation]:
            if obj is not None and hasattr(obj, "before_reset"):
                obj.before_reset()

    """------------------------------------------- overwrite -------------------------------------------------"""

    def convert_to_world_coordinates(self, vector, origin):
        return super(EgoDeliveryRobot, self).convert_to_world_coordinates([-vector[-1], vector[0]], origin)

    def convert_to_local_coordinates(self, vector, origin):
        ret = super(EgoDeliveryRobot, self).convert_to_local_coordinates(vector, origin)
        return np.array([ret[1], -ret[0]])

    @property
    def heading_theta(self):
        return wrap_to_pi(super(EgoDeliveryRobot, self).heading_theta + np.pi / 2)

    def set_heading_theta(self, heading_theta, in_rad=True) -> None:
        """
        Set heading theta for this object. Vehicle local frame has a 90 degree offset
        :param heading_theta: float in rad
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """
        super(EgoDeliveryRobot, self).set_heading_theta(heading_theta - np.pi / 2, in_rad)
        self.last_heading_dir = self.heading

    @property
    def roll(self):
        """
        Return the roll of this object
        """
        return np.deg2rad(self.origin.getR())

    def set_roll(self, roll):
        self.origin.setR(roll)

    @property
    def pitch(self):
        """
        Return the pitch of this object
        """
        return np.deg2rad(self.origin.getP())

    def set_pitch(self, pitch):
        self.origin.setP(pitch)

    def show_coordinates(self):
        if self.coordinates_debug_np is not None:
            self.coordinates_debug_np.reparentTo(self.origin)
            return
        height = self.HEIGHT + 0.2
        self.coordinates_debug_np = NodePath("debug coordinate")
        self.coordinates_debug_np.hide(CamMask.AllOn)
        self.coordinates_debug_np.show(CamMask.MainCam)
        # 90 degrees offset
        x = self.engine._draw_line_3d([0, 0, height], [0, 2, height], [1, 1, 1, 1], 3)
        y = self.engine._draw_line_3d([0, 0, height], [-1, 0, height], [1, 1, 1, 1], 3)
        z = self.engine._draw_line_3d([0, 0, height], [0, 0, height + 0.5], [1, 1, 1, 1], 3)
        x.reparentTo(self.coordinates_debug_np)
        y.reparentTo(self.coordinates_debug_np)
        z.reparentTo(self.coordinates_debug_np)
        self.coordinates_debug_np.reparentTo(self.origin)

    @property
    def lidar(self):
        return self.engine.get_sensor("lidar")

    @property
    def side_detector(self):
        return self.engine.get_sensor("side_detector")

    @property
    def lane_line_detector(self):
        return self.engine.get_sensor("lane_line_detector")
