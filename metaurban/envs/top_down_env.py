from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.obs.top_down_obs import TopDownObservation
from metaurban.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metaurban.utils import Config


class TopDownSingleFrameMetaUrbanEnv(SidewalkStaticMetaUrbanEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = SidewalkStaticMetaUrbanEnv.default_config()
        # config["vehicle_config"]["lidar"].update({"num_lasers": 0, "distance": 0})  # Remove lidar
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 84,
                "distance": 30
            }
        )
        return config

    def get_single_observation(self, _=None):
        return TopDownObservation(
            self.config["vehicle_config"],
            self.config["norm_pixel"],
            onscreen=self.config["use_render"],
            max_distance=self.config["distance"]
        )


class TopDownMetaUrban(TopDownSingleFrameMetaUrbanEnv):
    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )


class TopDownMetaUrbanEnvV2(SidewalkStaticMetaUrbanEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = SidewalkStaticMetaUrbanEnv.default_config()
        config["vehicle_config"]["lidar"] = {"num_lasers": 0, "distance": 0}  # Remove lidar
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 84,
                "distance": 30
            }
        )
        return config

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["use_render"],
            self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )