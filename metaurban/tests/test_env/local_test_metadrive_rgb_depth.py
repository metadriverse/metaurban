import numpy as np

from metaurban.constants import TerminationState
from metaurban.envs.metaurban_env import metaurbanEnv

info_keys = [
    "cost", "velocity", "steering", "acceleration", "step_reward", TerminationState.CRASH_VEHICLE,
    TerminationState.OUT_OF_ROAD, TerminationState.SUCCESS
]


def _act(env, action):
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isscalar(reward)
    assert isinstance(info, dict)
    for k in info_keys:
        assert k in info


def test_metaurban_env_rgb():
    env = metaurbanEnv(dict(image_observation=True))
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()


if __name__ == '__main__':
    test_metaurban_env_rgb()