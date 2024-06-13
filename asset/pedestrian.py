# Test Pedestrain manager that can generate static pedestrian on sidewalk
# Please refer to  metaurban.manager.sidewalk_manager for implementation detail
# !!!!!!!!!!!!You need to change asset used in  metaurban.manager.sidewalk_manager
from metaurban.component.traffic_participants.pedestrian import Pedestrian
from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.component.static_object.test_new_object import TestObject
from metaurban.envs.test_pede_metaurban_env import TestPedemetaurbanEnv

def try_pedestrian(render=False):
    env = TestPedemetaurbanEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 1.,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "debug": False,
            "cull_scene": False,
            "manual_control": True,
            "use_render": render,
            "decision_repeat": 5,
            "need_inverse_traffic": False,
            "rgb_clip": True,
            "map": "SX",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            # "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": False,
            "window_size": (2400, 1600),
            "vehicle_config": {
                "enable_reverse": False,
            },
        }
    )
    asset_metainfo = {
        "length": 2,
        "width": 2,
        "height": 2,
        "filename": "car-3f699c7ce86c4ba1bad62a350766556f.glb",
        "CLASS_NAME": "06e459171a264e999b3763335403b719",
        "hshift": 0,
        "pos0": 0,
        "pos1": 0,
        "pos2": 0,
        "scale": 1
    }
    env.reset()
    try:
        # obj_1 = env.engine.spawn_object(TestObject, position=[30, -5], heading_theta=0, random_seed=1, force_spawn=True, asset_metainfo = asset_metainfo)
        for s in range(1, 1000):
            o, r, tm, tc, info = env.step([0, 0])

    finally:
        env.close()


if __name__ == "__main__":
    try_pedestrian(True)
