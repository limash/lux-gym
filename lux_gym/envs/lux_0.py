from abc import ABC

import gym
# from gym import spaces

from kaggle_environments import make

import lux_gym.envs.tools as tools
from lux_gym.envs.lux.game import Game


class LuxEnv(gym.Env, ABC):

    def __init__(self, debug=False, seed=None):
        self._debug = debug

        tools.units_actions_dict.clear()
        if seed is not None:
            self._env = make("lux_ai_2021", configuration={"seed": seed, "loglevel": 2}, debug=debug)
        else:
            self._env = make("lux_ai_2021", configuration={"loglevel": 2}, debug=debug)
        self._positions = (0, 1)
        # game states allow easier game info scratching
        self._first_player_game_state = None
        self._second_player_game_state = None

    @property
    def configuration(self):
        return self._env.configuration

    @property
    def game_states(self):
        return self._first_player_game_state, self._second_player_game_state

    def reset(self):
        # update state
        tools.units_actions_dict.clear()
        self._env.reset()
        # get an observation from a 'shared' state, which an agent can use
        observation1 = self._env._Environment__get_shared_state(self._positions[0]).observation
        self._first_player_game_state = Game()
        self._first_player_game_state._initialize(observation1["updates"])
        self._first_player_game_state.player_id = observation1.player
        self._first_player_game_state._update(observation1["updates"][2:])
        # self._first_player_game_state.id = observation1.player
        self._first_player_game_state.fix_iteration_order()

        observation2 = self._env._Environment__get_shared_state(self._positions[1]).observation
        self._second_player_game_state = Game()
        self._second_player_game_state._initialize(observation2["updates"])
        self._second_player_game_state.player_id = observation2.player
        self._second_player_game_state._update(observation2["updates"][2:])
        # self._second_player_game_state.id = observation2.player
        self._second_player_game_state.fix_iteration_order()

        return observation1, observation2

    def reset_process(self):
        obs1, obs2 = self.reset()
        observations = (obs1, obs2)

        first_player_obs = tools.get_separate_outputs(obs1, self._first_player_game_state)
        second_player_obs = tools.get_separate_outputs(obs2, self._second_player_game_state)
        processed_observations = (first_player_obs, second_player_obs)

        return observations, processed_observations

    def step(self, actions):
        states = self._env.step(actions)
        dones = [False if state.status == 'ACTIVE' else True for state in states]

        observation1 = self._env._Environment__get_shared_state(self._positions[0]).observation
        self._first_player_game_state._update(observation1["updates"])

        observation2 = self._env._Environment__get_shared_state(self._positions[1]).observation
        self._second_player_game_state._update(observation2["updates"])

        return dones, (observation1, observation2)

    def step_process(self, actions):
        dones, (obs1, obs2) = self.step(actions)
        observations = (obs1, obs2)

        # processed_observation is what a model should consume
        first_player_obs = tools.get_separate_outputs(obs1, self._first_player_game_state)
        second_player_obs = tools.get_separate_outputs(obs2, self._second_player_game_state)
        processed_observations = (first_player_obs, second_player_obs)

        return dones, observations, processed_observations
