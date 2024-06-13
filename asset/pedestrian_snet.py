# Test sidewalk manager that can spawn all types of object on, near, or outside sidewalk
# Please refer to  metaurban.manager.sidewalk_manager for implementation detail
#
from metaurban.engine.asset_loader import AssetLoader
from metaurban.policy.replay_policy import ReplayEgoCarPolicy
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.instance_camera import InstanceCamera
asset_path = AssetLoader.asset_path
use_waymo = False
def try_pedestrian(render=False):
    env_config = {
        "sequential_seed": True,
        "reactive_traffic": True,
        "use_render": True,
        "data_directory": AssetLoader.file_path(
            asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
        ),
        "num_scenarios": 3 if use_waymo else 10,
        "agent_policy": ReplayEgoCarPolicy,
        "sensors": dict(
            rgb=(RGBCamera, 960, 640),
            instance=(InstanceCamera, 960, 640)
        )
    }
    from metaurban.envs.scenario_env import ScenarioEnv, ScenarioDiverseEnv
    env = ScenarioDiverseEnv(env_config)
    env.reset()
    try:
        # obj_1 = env.engine.spawn_object(TestObject, position=[30, -5], heading_theta=0, random_seed=1, force_spawn=True, asset_metainfo = asset_metainfo)
        for s in range(1, 100000000):
            o, r, tm, tc, info = env.step([0, 0])
            # for obj_id,obj in env.engine.get_objects().items():
            #     if isinstance(obj,CustomizedCar) or isinstance(obj, TestObject):
            #         print(obj.get_asset_metainfo())
            #     else:
            #         print(type(obj))
            ego = env.vehicle
            print(ego.crashed_objects)

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True

    finally:
        env.close()


if __name__ == "__main__":
    try_pedestrian(True)
