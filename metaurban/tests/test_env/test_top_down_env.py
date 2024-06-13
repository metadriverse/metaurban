import numpy as np
from metaurban.envs.scenario_env import ScenarioEnv, AssetLoader
from metaurban.envs.top_down_env import TopDownSingleFramemetaurbanEnv, TopDownmetaurban, TopDownmetaurbanEnvV2


def test_top_down_rendering():
    for env in [
            TopDownSingleFramemetaurbanEnv(dict(num_scenarios=5, map="C", traffic_density=1.0)),
            TopDownmetaurban(dict(num_scenarios=5, map="C", traffic_density=1.0)),
            TopDownmetaurban(dict(num_scenarios=5, map="C", frame_stack=1, post_stack=2)),
            TopDownmetaurbanEnvV2(dict(num_scenarios=5, map="C", frame_stack=1, post_stack=2)),
            ScenarioEnv(dict(
                num_scenarios=1,
                start_scenario_index=0,
                data_directory=AssetLoader.file_path("waymo", unix_style=False),
            )),
            ScenarioEnv(dict(
                num_scenarios=1,
                start_scenario_index=1,
                data_directory=AssetLoader.file_path("waymo", unix_style=False),
            )),
            ScenarioEnv(dict(
                num_scenarios=1,
                start_scenario_index=2,
                data_directory=AssetLoader.file_path("waymo", unix_style=False),
            )),
    ]:
        try:
            for _ in range(5):
                o, _ = env.reset()
                assert np.mean(o) > 0.0
                for _ in range(10):
                    o, *_ = env.step([0, 1])
                    assert np.mean(o) > 0.0
                for _ in range(10):
                    o, *_ = env.step([-0.05, 1])
                    assert np.mean(o) > 0.0
        finally:
            env.close()


def _vis_top_down_with_panda_render():
    env = TopDownmetaurban(dict(use_render=True))
    try:
        o, _ = env.reset()
        for i in range(1000):
            o, r, tm, tc, i = env.step([0, 1])
            if tm or tc:
                break
    finally:
        env.close()


def _vis_top_down_with_panda_render_and_top_down_visualization():
    env = TopDownmetaurban({"use_render": True})
    try:
        o, _ = env.reset()
        for i in range(2000):
            o, r, tm, tc, i = env.step([0, 1])
            if tm or tc:
                break
            env.render(mode="top_down")
    finally:
        env.close()


if __name__ == "__main__":
    # test_top_down_rendering()
    # _vis_top_down_with_panda_render()
    _vis_top_down_with_panda_render_and_top_down_visualization()
