from abc import ABC
# from collections import deque

import numpy as np
import gym
# from gym import spaces

from kaggle_environments import make

# from lux_gym.envs.lux.game import Game


class LuxEnv(gym.Env, ABC):

    def __init__(self, debug=False):
        self._debug = debug

        self._env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2}, debug=True)
        self._positions = (0, 1)

    def configuration(self):
        return self._env.configuration

    def reset(self):
        # update state
        self._env.reset()
        # get a 'shared' state, which an agent can use
        shared_state1 = self._env._Environment__get_shared_state(self._positions[0]).observation
        shared_state2 = self._env._Environment__get_shared_state(self._positions[1]).observation
        return shared_state1, shared_state2

    def step(self, actions):
        states = self._env.step(actions)
        dones = [False if state.status == 'ACTIVE' else True for state in states]
        shared_state1 = self._env._Environment__get_shared_state(self._positions[0]).observation
        shared_state2 = self._env._Environment__get_shared_state(self._positions[1]).observation
        return dones, (shared_state1, shared_state2)


def to_binary(d, m=8):
    """
    Args:
        d: is an array of decimal numbers to convert to binary
        m: is a number of positions in a binary number, 8 is enough for up to 256 decimal, 256 is 2^8
    Returns:
        np.ndarray of binary representation of d

    """
    reversed_order = ((d[:, None] & (1 << np.arange(m))) > 0).astype(np.uint8)
    return np.fliplr(reversed_order)
