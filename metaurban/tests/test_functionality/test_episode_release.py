from metaurban.envs.safe_metaurban_env import SafemetaurbanEnv


def test_episode_release():
    try:
        env = SafemetaurbanEnv(
            {
                "use_render": False,
                "num_scenarios": 100,
                "accident_prob": .8,
                "traffic_density": 0.5,
                "debug": True
            }
        )
        o, _ = env.reset()
        for i in range(1, 10):
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.reset()
            env.reset()
    finally:
        env.close()
