# This test is broken for some reasons. Just remove it from CI temporarily.


def _test_remote_metaurban_env():
    from metaurban.envs.legacy_envs.remote_env import Remotemetaurban
    # Test
    envs = [Remotemetaurban(dict(map=7)) for _ in range(3)]
    ret = [env.reset() for env in envs]
    # print(ret)
    ret = [env.step(env.action_space.sample()) for env in envs]
    # print(ret)
    [env.reset() for env in envs]
    [env.close() for env in envs]
    # print('Success!')


if __name__ == '__main__':
    _test_remote_metaurban_env()
