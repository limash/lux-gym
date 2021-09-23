from lux_gym.envs.lux.game import Game
from lux_gym.agents import half_imitator
import lux_gym.envs.tools as tools

game_state = None


def get_policy(name):
    policies = {
                "half_imitator": half_imitator.get_policy(),
                }
    return policies[name]


def get_agent(policy_name, is_gym=True):
    """
    The agents, which return actions only.

    Args:
        policy_name: valid names are "simple_rb", "ilia_rb", "nathan_rb"
        is_gym: if False, returns a valid for submission agent,
                if True, returns a gym agent that needs to know a game state
    Returns:
        agent or gym_agent
    """

    policy = get_policy(policy_name)

    def agent(observation, configuration):
        """This agent is valid for submission."""
        global game_state

        # Do not edit #
        if observation["step"] == 0:
            game_state = Game()
            game_state._initialize(observation["updates"])
            game_state.player_id = observation.player
            game_state._update(observation["updates"][2:])
            # game_state.id = observation.player
            game_state.fix_iteration_order()
        else:
            game_state._update(observation["updates"])

        actions, actions_dict, actions_probs_dict, processed_observations = policy(game_state, observation)
        return actions

    def gym_agent(observation, configuration, current_game_state):
        """This agent is valid for a gym environment with several players."""

        actions, actions_dict, actions_probs_dict, processed_observations = policy(current_game_state, observation)
        return actions

    if is_gym:
        return gym_agent
    else:
        return agent


def get_processing_agent(policy_name):
    """The agents, which return processed data with actions. For trajectories collection. """

    policy = get_policy(policy_name)

    def gym_agent(observation, configuration, current_game_state):
        """This agent is valid for a gym environment with several players."""

        actions, actions_dict, actions_probs_dict, processed_observations = policy(current_game_state, observation)
        if processed_observations is None:
            processed_observations = tools.get_separate_outputs(observation, current_game_state)

        # for act_values, obs_values in zip(actions_dict.values(), processed_observations.values()):
        #     if len(act_values) != len(obs_values):
        #         raise ValueError

        return actions, actions_dict, actions_probs_dict, processed_observations, observation.reward

    return gym_agent
