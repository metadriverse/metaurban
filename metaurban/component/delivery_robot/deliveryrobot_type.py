from panda3d.core import Filename
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp
from panda3d.core import Material, Vec3, TransformState
from metaurban.engine.asset_loader import AssetLoader
from metaurban.component.pg_space import ParameterSpace, VehicleParameterSpace
from metaurban.component.delivery_robot.base_deliveryrobot import BaseDeliveryRobot, EgoDeliveryRobot
import platform
from metaurban.constants import Semantics, MetaUrbanType
from metaurban.engine.engine_utils import get_engine, engine_initialized
from metaurban.engine.physics_node import BaseRigidBodyNode
from metaurban.utils import Config
from typing import Union, Optional
import os
from metaurban.component.delivery_robot.vel_driven_deliveryrobot import EgoDeliveryRobot as EgoVelDeliveryRobot


def convert_path(pth):
    return Filename.from_os_specific(pth).get_fullpath()


class CustomizedCar(BaseDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
    TIRE_RADIUS = 0.3305  #0.313
    CHASSIS_TO_WHEEL_AXIS = 0.2
    TIRE_WIDTH = 0.255  #0.25
    MASS = 1595  #1100
    LATERAL_TIRE_TO_CENTER = 1  #0.815
    FRONT_WHEELBASE = 1.36  #1.05234
    REAR_WHEELBASE = 1.45  #1.4166
    #path = ['ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]
    path = ['lambo/vehicle.glb', (0.5, 0.5, 0.5), (1.09, 0, 0.6), (0, 0, 0)]

    def __init__(
        self,
        test_asset_meta_info: dict,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
        position=None,
        heading=None
    ):
        # print("init!")
        self.asset_meta_info = test_asset_meta_info
        self.update_asset_metainfo(test_asset_meta_info)
        super().__init__(vehicle_config, name, random_seed, position, heading)

    def get_asset_metainfo(self):
        return self.asset_meta_info

    @classmethod
    def update_asset_metainfo(cls, asset_metainfo: dict):
        # print(asset_metainfo)
        cls.PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
        cls.TIRE_RADIUS = asset_metainfo["TIRE_RADIUS"]  # 0.313
        cls.TIRE_WIDTH = asset_metainfo["TIRE_WIDTH"]  # 0.25
        cls.MASS = asset_metainfo["MASS"]  # 1100
        cls.LATERAL_TIRE_TO_CENTER = asset_metainfo["LATERAL_TIRE_TO_CENTER"]  # 0.815
        cls.FRONT_WHEELBASE = asset_metainfo["FRONT_WHEELBASE"]  # 1.05234
        cls.REAR_WHEELBASE = asset_metainfo["REAR_WHEELBASE"]  # 1.4166
        # path = ['ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]
        cls.path = [
            asset_metainfo["MODEL_PATH"],
            tuple(asset_metainfo["MODEL_SCALE"]),
            tuple(asset_metainfo["MODEL_OFFSET"]),
            tuple(asset_metainfo["MODEL_HPR"])
        ]
        cls.LENGTH = asset_metainfo["LENGTH"]
        cls.HEIGHT = asset_metainfo["HEIGHT"]
        cls.WIDTH = asset_metainfo["WIDTH"]
        # print(asset_metainfo)
        if "CHASSIS_TO_WHEEL_AXIS" not in asset_metainfo:
            asset_metainfo["CHASSIS_TO_WHEEL_AXIS"] = 0.2
        if "TIRE_SCALE" not in asset_metainfo:
            asset_metainfo["TIRE_SCALE"] = 1
        if "TIRE_OFFSET" not in asset_metainfo:
            asset_metainfo["TIRE_OFFSET"] = 0
        cls.CHASSIS_TO_WHEEL_AXIS = asset_metainfo["CHASSIS_TO_WHEEL_AXIS"]
        cls.TIRE_SCALE = asset_metainfo["TIRE_SCALE"]
        cls.TIRE_OFFSET = asset_metainfo["TIRE_OFFSET"]

    def _create_vehicle_chassis(self):
        # self.LENGTH = type(self).LENGTH
        # self.WIDTH = type(self).WIDTH
        # self.HEIGHT = type(self).HEIGHT

        # assert self.LENGTH < BaseDeliveryRobot.MAX_LENGTH, "Vehicle is too large!"
        # assert self.WIDTH < BaseDeliveryRobot.MAX_WIDTH, "Vehicle is too large!"

        chassis = BaseRigidBodyNode(self.name, MetaUrbanType.VEHICLE)
        self._node_path_list.append(chassis)

        chassis_shape = BulletBoxShape(
            Vec3(self.WIDTH / 2, self.LENGTH / 2, (self.HEIGHT - self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS) / 2)
        )
        ts = TransformState.makePos(Vec3(0, 0, self.TIRE_RADIUS + self.CHASSIS_TO_WHEEL_AXIS))
        # ts = TransformState.makePos(Vec3(0, 0, self.TIRE_RADIUS))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        self.dynamic_nodes.append(vehicle_chassis)
        return vehicle_chassis

    @classmethod
    def LENGTH(cls):
        return cls.LENGTH

    @classmethod
    def HEIGHT(cls):
        return cls.HEIGHT

    @classmethod
    def WIDTH(cls):
        return cls.WIDTH

    def _add_visualization(self):
        if self.render:
            [path, scale, offset, HPR] = self.path
            car_model = self.loader.loadModel(AssetLoader.file_path("models", path))
            car_model.setTwoSided(False)
            BaseDeliveryRobot.model_collection[path] = car_model
            car_model.setScale(scale)
            # model default, face to y
            car_model.setHpr(*HPR)
            car_model.setPos(offset[0], offset[1], offset[-1])
            # car_model.setZ(-self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS + offset[-1])
            car_model.instanceTo(self.origin)
            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (
                        self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.2
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
        axis_height = self.TIRE_OFFSET
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

        if self.render:
            model = 'right_tire_front.gltf' if front else 'right_tire_back.gltf'
            model_path = AssetLoader.file_path("models", os.path.dirname(self.path[0]), model)
            wheel_model = self.loader.loadModel(model_path)
            wheel_model.setTwoSided(self.TIRE_TWO_SIDED)
            wheel_model.reparentTo(wheel_np)
            wheel_model.set_scale(
                1 * self.TIRE_SCALE * self.TIRE_MODEL_CORRECT if left else -1 * self.TIRE_SCALE *
                self.TIRE_MODEL_CORRECT
            )
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


class EgoVehicle(EgoDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.ROBOT_)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    # LATERAL_TIRE_TO_CENTER = 0.7
    # TIRE_TWO_SIDED = True
    # FRONT_WHEELBASE = 1.385  # 0.3#0.3#0.5#0.3#0.5#0.25#1.385
    # REAR_WHEELBASE = 1.11  #0.3#0.3#0.5#0.3#0.5#0.25#1.11
    # TIRE_RADIUS = 0.376  #0.2#0.1#0.2#0.05#0.15#0.1#0.376
    # TIRE_WIDTH = 0.25  #0.2#0.1#0.2#0.05#0.1#0.1#0.25
    # MASS = 500
    TIRE_RADIUS = 0.3305  #0.313
    CHASSIS_TO_WHEEL_AXIS = 0.2
    TIRE_WIDTH = 0.255  #0.25
    MASS = 1595  #1100
    LATERAL_TIRE_TO_CENTER = 1  #0.815
    FRONT_WHEELBASE = 1.36  #1.05234
    REAR_WHEELBASE = 1.45  #1.4166
    LIGHT_POSITION = (-0.57, 1.86, 0.23)
    path = ['coco-gradient.glb', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 1.5  # meters

    @property
    def HEIGHT(self):
        return 0.9  # meters

    @property
    def WIDTH(self):
        return 1.0  # meters

    @property
    def RADIUS(self):
        return 1.5
    
class EgoVelVehicle(EgoVelDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.ROBOT_)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    # LATERAL_TIRE_TO_CENTER = 0.7
    # TIRE_TWO_SIDED = True
    # FRONT_WHEELBASE = 1.385  # 0.3#0.3#0.5#0.3#0.5#0.25#1.385
    # REAR_WHEELBASE = 1.11  #0.3#0.3#0.5#0.3#0.5#0.25#1.11
    # TIRE_RADIUS = 0.376  #0.2#0.1#0.2#0.05#0.15#0.1#0.376
    # TIRE_WIDTH = 0.25  #0.2#0.1#0.2#0.05#0.1#0.1#0.25
    # MASS = 500
    TIRE_RADIUS = 0.3305  #0.313
    CHASSIS_TO_WHEEL_AXIS = 0.2
    TIRE_WIDTH = 0.255  #0.25
    MASS = 1595  #1100
    LATERAL_TIRE_TO_CENTER = 1  #0.815
    FRONT_WHEELBASE = 1.36  #1.05234
    REAR_WHEELBASE = 1.45  #1.4166
    LIGHT_POSITION = (-0.57, 1.86, 0.23)
    path = ['coco-gradient.glb', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 1.5  # meters

    @property
    def HEIGHT(self):
        return 0.9  # meters

    @property
    def WIDTH(self):
        return 1.0  # meters

    @property
    def RADIUS(self):
        return 1.5

class EgoWheelchair(EgoDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.ROBOT_)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    # LATERAL_TIRE_TO_CENTER = 0.7
    # TIRE_TWO_SIDED = True
    # FRONT_WHEELBASE = 1.385  # 0.3#0.3#0.5#0.3#0.5#0.25#1.385
    # REAR_WHEELBASE = 1.11  #0.3#0.3#0.5#0.3#0.5#0.25#1.11
    # TIRE_RADIUS = 0.376  #0.2#0.1#0.2#0.05#0.15#0.1#0.376
    # TIRE_WIDTH = 0.25  #0.2#0.1#0.2#0.05#0.1#0.1#0.25
    # MASS = 500
    TIRE_RADIUS = 0.3305  #0.313
    CHASSIS_TO_WHEEL_AXIS = 0.2
    TIRE_WIDTH = 0.255  #0.25
    MASS = 1595  #1100
    LATERAL_TIRE_TO_CENTER = 1  #0.815
    FRONT_WHEELBASE = 1.36  #1.05234
    REAR_WHEELBASE = 1.45  #1.4166
    LIGHT_POSITION = (-0.57, 1.86, 0.23)
    path = ['../../assets_pedestrain/special_agents/free3DVersion.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 1.5  # meters

    @property
    def HEIGHT(self):
        return 0.9  # meters

    @property
    def WIDTH(self):
        return 1.0  # meters

    @property
    def RADIUS(self):
        return 1.5


class DefaultVehicle(BaseDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.ROBOT_)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    LATERAL_TIRE_TO_CENTER = 0.7
    TIRE_TWO_SIDED = True
    FRONT_WHEELBASE = 1.385
    REAR_WHEELBASE = 1.11
    TIRE_RADIUS = 0.376
    TIRE_WIDTH = 0.25
    MASS = 800
    LIGHT_POSITION = (-0.57, 1.86, 0.23)
    path = ['deliveryrobot-default.glb', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 1.5  # meters

    @property
    def HEIGHT(self):
        return 0.8  # meters

    @property
    def WIDTH(self):
        return 1.0  # meters

    @property
    def RADIUS(self):
        return 1.5


# When using DefaultVehicle as traffic, please use this class.


class TrafficDefaultVehicle(DefaultVehicle):
    pass


class StaticDefaultVehicle(DefaultVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.STATIC_DEFAULT_VEHICLE)


class XLVehicle(BaseDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.XL_VEHICLE)
    # LENGTH = 5.8
    # WIDTH = 2.3
    # HEIGHT = 2.8
    TIRE_RADIUS = 0.37
    TIRE_MODEL_CORRECT = -1
    REAR_WHEELBASE = 1.075
    FRONT_WHEELBASE = 1.726
    LATERAL_TIRE_TO_CENTER = 0.931
    CHASSIS_TO_WHEEL_AXIS = 0.3
    TIRE_WIDTH = 0.5
    MASS = 1600
    LIGHT_POSITION = (-0.75, 2.7, 0.2)
    SEMANTIC_LABEL = Semantics.TRUCK.label
    path = ['truck/vehicle.gltf', (1, 1, 1), (0, 0.25, 0.04), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 5.74  # meters

    @property
    def HEIGHT(self):
        return 2.8  # meters

    @property
    def WIDTH(self):
        return 2.3  # meters


class LVehicle(BaseDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.L_VEHICLE)
    # LENGTH = 4.5
    # WIDTH = 1.86
    # HEIGHT = 1.85
    TIRE_RADIUS = 0.429
    REAR_WHEELBASE = 1.218261
    FRONT_WHEELBASE = 1.5301
    LATERAL_TIRE_TO_CENTER = 0.75
    TIRE_WIDTH = 0.35
    MASS = 1300
    LIGHT_POSITION = (-0.65, 2.13, 0.3)

    path = ['lada/vehicle.gltf', (1.1, 1.1, 1.1), (0, -0.27, 0.07), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.87  # meters

    @property
    def HEIGHT(self):
        return 1.85  # meters

    @property
    def WIDTH(self):
        return 2.046  # meters


class MVehicle(BaseDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
    # LENGTH = 4.4
    # WIDTH = 1.85
    # HEIGHT = 1.37
    TIRE_RADIUS = 0.39
    REAR_WHEELBASE = 1.203
    FRONT_WHEELBASE = 1.285
    LATERAL_TIRE_TO_CENTER = 0.803
    TIRE_WIDTH = 0.3
    MASS = 1200
    LIGHT_POSITION = (-0.67, 1.86, 0.22)

    path = ['130/vehicle.gltf', (1, 1, 1), (0, -0.05, 0.1), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.6  # meters

    @property
    def HEIGHT(self):
        return 1.37  # meters

    @property
    def WIDTH(self):
        return 1.85  # meters


class SVehicle(BaseDeliveryRobot):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.S_VEHICLE)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    LATERAL_TIRE_TO_CENTER = 0.7
    TIRE_TWO_SIDED = True
    FRONT_WHEELBASE = 1.385
    REAR_WHEELBASE = 1.11
    TIRE_RADIUS = 0.376
    TIRE_WIDTH = 0.25
    MASS = 800
    LIGHT_POSITION = (-0.57, 1.86, 0.23)

    @property
    def path(self):
        if self.use_render_pipeline and platform.system() != "Linux":
            # vfs = VirtualFileSystem.get_global_ptr()
            # vfs.mount(convert_path(AssetLoader.file_path("models", "beetle")), "/$$beetle_model", 0)
            return ['beetle/vehicle.bam', (0.0077, 0.0077, 0.0077), (0.04512, -0.24 - 0.04512, 1.77), (-90, -90, 0)]
        else:
            factor = 1
            return ['beetle/vehicle.gltf', (factor, factor, factor), (0, -0.2, 0.03), (0, 0, 0)]

    @property
    def LENGTH(self):
        return 4.3  # meters

    @property
    def HEIGHT(self):
        return 1.70  # meters

    @property
    def WIDTH(self):
        return 1.70  # meters


class VaryingDynamicsVehicle(DefaultVehicle):
    @property
    def WIDTH(self):
        return self.config["width"] if self.config["width"] is not None else super(VaryingDynamicsVehicle, self).WIDTH

    @property
    def LENGTH(self):
        return self.config["length"] if self.config["length"] is not None else super(
            VaryingDynamicsVehicle, self
        ).LENGTH

    @property
    def HEIGHT(self):
        return self.config["height"] if self.config["height"] is not None else super(
            VaryingDynamicsVehicle, self
        ).HEIGHT

    @property
    def MASS(self):
        return self.config["mass"] if self.config["mass"] is not None else super(VaryingDynamicsVehicle, self).MASS

    def reset(
        self,
        random_seed=None,
        vehicle_config=None,
        position=None,
        heading: float = 0.0,  # In degree!
        *args,
        **kwargs
    ):

        assert "width" not in self.PARAMETER_SPACE
        assert "height" not in self.PARAMETER_SPACE
        assert "length" not in self.PARAMETER_SPACE
        should_force_reset = False
        if vehicle_config is not None:
            if vehicle_config["width"] is not None and vehicle_config["width"] != self.WIDTH:
                should_force_reset = True
            if vehicle_config["height"] is not None and vehicle_config["height"] != self.HEIGHT:
                should_force_reset = True
            if vehicle_config["length"] is not None and vehicle_config["length"] != self.LENGTH:
                should_force_reset = True
            if "max_engine_force" in vehicle_config and \
                    vehicle_config["max_engine_force"] is not None and \
                    vehicle_config["max_engine_force"] != self.config["max_engine_force"]:
                should_force_reset = True
            if "max_brake_force" in vehicle_config and \
                    vehicle_config["max_brake_force"] is not None and \
                    vehicle_config["max_brake_force"] != self.config["max_brake_force"]:
                should_force_reset = True
            if "wheel_friction" in vehicle_config and \
                    vehicle_config["wheel_friction"] is not None and \
                    vehicle_config["wheel_friction"] != self.config["wheel_friction"]:
                should_force_reset = True
            if "max_steering" in vehicle_config and \
                    vehicle_config["max_steering"] is not None and \
                    vehicle_config["max_steering"] != self.config["max_steering"]:
                self.max_steering = vehicle_config["max_steering"]
                should_force_reset = True
            if "mass" in vehicle_config and \
                    vehicle_config["mass"] is not None and \
                    vehicle_config["mass"] != self.config["mass"]:
                should_force_reset = True

        # def process_memory():
        #     import psutil
        #     import os
        #     process = psutil.Process(os.getpid())
        #     mem_info = process.memory_info()
        #     return mem_info.rss
        #
        # cm = process_memory()

        if should_force_reset:
            self.destroy()
            self.__init__(
                vehicle_config=vehicle_config,
                name=self.name,
                random_seed=self.random_seed,
                position=position,
                heading=heading,
                _calling_reset=False
            )

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format("1 Force Re-Init Vehicle", (lm - cm) / 1e6))
            # cm = lm

        assert self.max_steering == self.config["max_steering"]

        ret = super(VaryingDynamicsVehicle, self).reset(
            random_seed=random_seed, vehicle_config=vehicle_config, position=position, heading=heading, *args, **kwargs
        )

        # lm = process_memory()
        # print("{}:  Reset! Mem Change {:.3f}MB".format("2 Force Reset Vehicle", (lm - cm) / 1e6))
        # cm = lm

        return ret


def random_DeliveryRobot_type(np_random, p=None):
    prob = [1 / len(DeliveryRobot_type) for _ in range(len(DeliveryRobot_type))] if p is None else p
    return DeliveryRobot_type[np_random.choice(list(DeliveryRobot_type.keys()), p=prob)]


DeliveryRobot_type = {"default": DefaultVehicle, 'ego': EgoVehicle, 'wheelchair': EgoWheelchair, 'vel_ego': EgoVelVehicle}

DeliveryRobot_class_to_type = inv_map = {v: k for k, v in DeliveryRobot_type.items()}
