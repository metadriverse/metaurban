from metaurban.envs import metaurbanEnv


# Related issue:
# https://github.com/metaurbanrse/metaurban/issues/191
def test_close_and_reset():

    env = metaurbanEnv({"start_seed": 1000, "num_scenarios": 1})
    eval_env = metaurbanEnv()
    assert eval_env.action_space.contains(env.action_space.sample())
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
    env.close()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
    env.close()

    eval_env.reset()
    for i in range(100):
        eval_env.step(eval_env.action_space.sample())
    eval_env.close()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
    env.close()


if __name__ == '__main__':
    test_close_and_reset()
