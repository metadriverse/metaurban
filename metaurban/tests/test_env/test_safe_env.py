from metaurban.envs.safe_metaurban_env import SafemetaurbanEnv


def test_safe_env(vis=False):
    config = {"num_scenarios": 100, "start_seed": 75}
    if vis:
        config["vehicle_config"] = {"show_line_to_navi_mark": True}
        config["use_render"] = True
        config["manual_control"] = True
        config["controller"] = "steering_wheel"

    env = SafemetaurbanEnv(config)
    try:
        o, _ = env.reset()
        total_cost = 0
        for ep in range(5):
            for i in range(1, 100):
                o, r, tm, tc, info = env.step([0, 1])
                total_cost += info["cost"]
                assert env.observation_space.contains(o)
                if tm or tc:
                    total_cost = 0
                    # print("Reset")
                    env.reset()
        env.close()
    finally:
        env.close()


if __name__ == '__main__':
    test_safe_env(vis=True)
