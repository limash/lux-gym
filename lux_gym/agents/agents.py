from lux_gym.envs.lux.game import Game
from lux_gym.agents import simple_rb, ilia_rb, nathan_rb

game_state = None

policies = {"simple_rb": simple_rb,
            "ilia_rb": ilia_rb,
            "nathan_rb": nathan_rb}


def get_agent(policy_name, is_gym=True):
    """
    Args:
        policy_name: valid names are "simple_rb", "ilia_rb", "nathan_rb"
        is_gym: if False, returns a valid for submission agent,
                if True, returns a gym agent that needs to know a game state
    Returns:
        agent or gym_agent
    """

    policy = policies[policy_name].policy

    def agent(observation, configuration):
        """This agent is valid for submission."""
        global game_state

        # Do not edit #
        if observation["step"] == 0:
            game_state = Game()
            game_state._initialize(observation["updates"])
            game_state._update(observation["updates"][2:])
            game_state.id = observation.player
        else:
            game_state._update(observation["updates"])

        actions = policy(game_state, observation)
        return actions

    def gym_agent(observation, configuration, current_game_state):
        """This agent is valid for a gym environment with several players."""

        actions = policy(current_game_state, observation)
        return actions

    if is_gym:
        return gym_agent
    else:
        return agent
