
import gymnasium as gym
from metaurban.component.sensors.base_camera import BaseCamera
import numpy as np

from metaurban.component.vehicle.base_vehicle import BaseVehicle
from metaurban.obs.observation_base import BaseObservation
from metaurban.obs.state_obs import StateObservation

_cuda_enable = True
try:
    import cupy as cp
except ImportError:
    _cuda_enable = False


class ImageStateObservation(BaseObservation):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, config):
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])
        self.state_obs = StateObservation(config)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )

    def observe(self, vehicle: BaseVehicle):
        return {self.IMAGE: self.img_obs.observe(), self.STATE: self.state_obs.observe(vehicle)}

    def destroy(self):
        super(ImageStateObservation, self).destroy()
        self.img_obs.destroy()
        self.state_obs.destroy()
        
        
class ThreeSourceMixObservation(BaseObservation):
    IMAGE = "image"
    DEPTH = 'depth'
    STATE = "state"

    def __init__(self, config):
        super(ThreeSourceMixObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, 'rgb_camera', config["norm_pixel"])
        self.depth_obs = ImageObservation(config, 'depth_camera', config["norm_pixel"])
        self.state_obs = LidarStateObservation(config)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space,
                self.DEPTH: self.depth_obs.observation_space
            }
        )

    def observe(self, vehicle: BaseVehicle):
        return {self.IMAGE: self.img_obs.observe(), self.STATE: self.state_obs.observe(vehicle), self.DEPTH: self.depth_obs.observe()}

    def destroy(self):
        super(ThreeSourceMixObservation, self).destroy()
        self.img_obs.destroy()
        self.state_obs.destroy()
        self.depth_obs.destroy()


class ImageObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config, image_source: str, clip_rgb: bool):
        self.enable_cuda = config["image_on_cuda"]
        if self.enable_cuda:
            assert _cuda_enable, "CuPy is not enabled. Fail to set up image_on_cuda."
        self.STACK_SIZE = config["stack_size"]
        self.image_source = image_source
        super(ImageObservation, self).__init__(config)
        self.norm_pixel = clip_rgb
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32 if self.norm_pixel else np.uint8)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    @property
    def observation_space(self):
        sensor_cls = self.config["sensors"][self.image_source][0]
        assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        channel = sensor_cls.num_channels if sensor_cls != "MainCamera" else 3
        shape = (self.config["sensors"][self.image_source][2],
                 self.config["sensors"][self.image_source][1]) + (channel, self.STACK_SIZE)
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, new_parent_node=None, position=None, hpr=None):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """
        new_obs = self.engine.get_sensor(self.image_source).perceive(self.norm_pixel, new_parent_node, position, hpr)
        self.state = cp.roll(self.state, -1, axis=-1) if self.enable_cuda else np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = new_obs
        return self.state

    def get_image(self):
        return self.state.copy()[:, :, -1]

    def reset(self, env, vehicle=None):
        """
        Clear stack
        :param env: metaurban
        :param vehicle: BaseVehicle
        :return: None
        """
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    def destroy(self):
        """
        Clear memory
        """
        super(ImageObservation, self).destroy()
        self.state = None


class LidarStateObservation(BaseObservation):
    """
    This observation uses lidar to detect moving objects
    """
    def __init__(self, config):
        self.state_obs = StateObservation(config)
        super(LidarStateObservation, self).__init__(config)
        self.cloud_points = None
        self.detected_objects = None

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["vehicle_config"]["lidar"]["num_lasers"] > 0 and self.config["vehicle_config"]["lidar"][
                "distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["vehicle_config"]["lidar"][
                "num_lasers"] + self.config["vehicle_config"]["lidar"]["num_others"] * 4
            if self.config["vehicle_config"]["lidar"]["add_others_navi"]:
                lidar_dim += self.config["vehicle_config"]["lidar"]["num_others"] * 4
            shape[0] += lidar_dim
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        """
        State observation + Navi info + 4 * closest vehicle info + Lidar points ,
        Definition of State Observation and Navi information can be found in **class StateObservation**
        Other vehicles' info: [
                              Projection of distance between ego and another vehicle on ego vehicle's heading direction,
                              Projection of distance between ego and another vehicle on ego vehicle's side direction,
                              Projection of speed between ego and another vehicle on ego vehicle's heading direction,
                              Projection of speed between ego and another vehicle on ego vehicle's side direction,
                              ] * 4, dim = 16

        Lidar points: 240 lidar points surrounding vehicle, starting from the vehicle head in clockwise direction

        :param vehicle: BaseVehicle
        :return: observation in 9 + 10 + 16 + 240 dim
        """
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
        ret = self.current_observation
        return ret.astype(np.float32)

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def lidar_observe(self, vehicle):
        other_v_info = []
        if vehicle.config["lidar"]["num_lasers"] > 0 and vehicle.config["lidar"]["distance"] > 0:
            cloud_points, detected_objects = self.engine.get_sensor("lidar").perceive(
                vehicle,
                physics_world=self.engine.physics_world.dynamic_world,
                num_lasers=vehicle.config["lidar"]["num_lasers"],
                distance=vehicle.config["lidar"]["distance"],
                show=vehicle.config["show_lidar"],
            )
            if vehicle.config["lidar"]["num_others"] > 0:
                other_v_info += self.engine.get_sensor("lidar").get_surrounding_vehicles_info(
                    vehicle, detected_objects, vehicle.config["lidar"]["distance"],
                    vehicle.config["lidar"]["num_others"], vehicle.config["lidar"]["add_others_navi"]
                )
            other_v_info += self._add_noise_to_cloud_points(
                cloud_points,
                gaussian_noise=vehicle.config["lidar"]["gaussian_noise"],
                dropout_prob=vehicle.config["lidar"]["dropout_prob"]
            )
            self.cloud_points = cloud_points
            self.detected_objects = detected_objects
        return other_v_info

    def _add_noise_to_cloud_points(self, points, gaussian_noise, dropout_prob):
        if gaussian_noise > 0.0:
            points = np.asarray(points)
            points = np.clip(points + np.random.normal(loc=0.0, scale=gaussian_noise, size=points.shape), 0.0, 1.0)

        if dropout_prob > 0.0:
            assert dropout_prob <= 1.0
            points = np.asarray(points)
            points[np.random.uniform(0, 1, size=points.shape) < dropout_prob] = 0.0

        return list(points)

    def destroy(self):
        """
        Clear allocated memory
        """
        self.state_obs.destroy()
        super(LidarStateObservation, self).destroy()
        self.cloud_points = None
        self.detected_objects = None