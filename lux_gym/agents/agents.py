from lux_gym.envs.lux.game import Game
from lux_gym.agents import half_imitator_shrub, half_imitator_six_actions, half_imitator_six_actions_eff
from lux_gym.agents import title_agent
from lux_gym.agents.compare_agent_1 import compare_agent as compare_agent1
from lux_gym.agents.compare_agent_2 import compare_agent as compare_agent2
import lux_gym.envs.tools as tools

game_state = None


policies = {
            "actor_critic_residual_shrub": half_imitator_shrub.get_policy,
            "actor_critic_residual_six_actions": half_imitator_six_actions.get_policy,
            "actor_critic_efficient_six_actions": half_imitator_six_actions_eff.get_policy,
            "title_agent": title_agent.get_policy,
            "compare_agent": compare_agent1.get_policy,
            "compare_agent_eff": compare_agent2.get_policy,
            }


def get_agent(policy_name, data=None, is_gym=True):
    """
    The agents, which return actions only.

    Args:
        policy_name: valid names are "simple_rb", "ilia_rb", "nathan_rb"
        data: weights
        is_gym: if False, returns a valid for submission agent,
                if True, returns a gym agent that needs to know a game state
    Returns:
        agent or gym_agent
    """

    policy = policies[policy_name](data)

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


def get_processing_agent(policy_name, data=None):
    """The agents, which return processed data with actions. For trajectories collection. """

    policy = policies[policy_name](data)

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
