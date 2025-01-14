from metaurban.envs.metaurban_env import metaurbanEnv

if __name__ == "__main__":
    config = {
        "num_scenarios": 10,
        "traffic_density": .0,
        # "use_render":True,
        "map": "SSSSS",
        # "manual_control":True,
        "controller": "steering_wheel",
        "random_agent_model": False,
        "vehicle_config": {
            "vehicle_model": "default",
            # "vehicle_model":"s",
            # "vehicle_model":"m",
            # "vehicle_model":"l",
        }
    }
    env = metaurbanEnv(config)
    import time

    start = time.time()
    o, _ = env.reset()
    a = [.0, 1.]
    for s in range(1, 100000):
        o, r, tm, tc, info = env.step(a)
        if env.agent.speed_km_h > 100:
            a = [0, -1]
            # print("0-100 km/h acc use time:{}".format(s * 0.1))
            pre_pos = env.agent.position[0]
        if a == [0, -1] and env.agent.speed_km_h < 1:
            # print("0-100 brake use dist:{}".format(env.agent.position[0] - pre_pos))
            break
    env.close()
