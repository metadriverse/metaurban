from metaurban.engine.asset_loader import AssetLoader
from metaurban.scenario.utils import draw_map
from metaurban.envs.metaurban_env import metaurbanEnv
from metaurban.envs.scenario_env import ScenarioEnv
from metaurban.policy.idm_policy import TrajectoryIDMPolicy


def test_export_waymo_map(render=False):
    TrajectoryIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    env = ScenarioEnv(
        {
            "manual_control": False,
            "no_traffic": True,
            "use_render": False,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", unix_style=False),
            "num_scenarios": 3
        }
    )
    try:
        for seed in range(3):
            env.reset(seed=seed)
            map_vector = env.current_map.get_map_features()
            draw_map(map_vector, True if render else False)
    finally:
        env.close()


def test_metaurban_map_export(render=False):
    env = metaurbanEnv(dict(image_observation=False, map=6, num_scenarios=1, start_seed=0))
    try:
        env.reset(seed=0)
        map_vector = env.current_map.get_map_features()
        draw_map(map_vector, True if render else False)
    finally:
        env.close()


if __name__ == "__main__":
    test_export_waymo_map(True)
