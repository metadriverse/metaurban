"""Dynamic env smoke: real pedestrian actors spawn, survive resets, and zero-spawn works.

Needs the pedestrian asset pack at metaurban/assets_pedestrain/ (pull_asset.py).
Run: python -m metaurban.tests.test_dynamic_env_pedestrians
"""
from metaurban import SidewalkDynamicMetaUrbanEnv
from metaurban.component.agents.pedestrian.base_pedestrian import BasePedestrian

BASE = dict(
    object_density=0.1,
    use_render=False,
    map='X',
    horizon=100,
    drivable_area_extension=55,
    height_scale=1,
    crswalk_density=1,
    walk_on_all_regions=False,
    spawn_wheelchairman_num=0,
    spawn_edog_num=0,
    spawn_erobot_num=0,
    spawn_drobot_num=0,
)


def test_pedestrians_across_resets():
    env = SidewalkDynamicMetaUrbanEnv(dict(BASE, num_scenarios=2, spawn_human_num=5, max_actor_num=5))
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            for _ in range(120):
                _, _, tm, tc, _ = env.step([0.0, 0.0])
                if tm or tc:
                    break
            mgr = env.engine.humanoid_manager
            scene = {o.name for o in env.engine.get_objects().values() if isinstance(o, BasePedestrian)}
            assert len(scene) == 5, scene
            # reset() must drop last episode's humanoids or trajectories de-sync
            assert sorted(v.name for v in mgr._traffic_humanoids) == sorted(scene)
    finally:
        env.close()


def test_zero_spawn():
    env = SidewalkDynamicMetaUrbanEnv(dict(BASE, num_scenarios=1, spawn_human_num=0, max_actor_num=0))
    try:
        env.reset(seed=0)
        for _ in range(10):
            env.step([0.0, 0.0])
    finally:
        env.close()


if __name__ == "__main__":
    test_pedestrians_across_resets()
    test_zero_spawn()
    print("dynamic env pedestrian smoke passed")
