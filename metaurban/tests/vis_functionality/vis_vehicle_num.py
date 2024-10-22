from metaurban.envs.metaurban_env import metaurbanEnv

if __name__ == "__main__":
    env = metaurbanEnv({
        "num_scenarios": 100,
        "start_seed": 5000,
        "traffic_density": 0.08,
    })
    env.reset()
    count = []
    for i in range(1, 101):
        o, r, tm, tc, info = env.step([0, 1])
        env.reset()
        # print(
        #     "Current map {}, vehicle number {}.".format(env.current_seed, env.engine.traffic_manager.get_vehicle_num())
        # )
        count.append(env.engine.traffic_manager.get_vehicle_num())
    # print(min(count), sum(count) / len(count), max(count))
    env.close()
