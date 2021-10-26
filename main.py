import gym
import kaggle_environments as kaggle

import lux_gym.agents.agents as agents


def show_gym(number_of_iterations, agent):
    env = gym.make('lux_gym:lux-v0', debug=True)
    configuration = env.configuration
    for i in range(number_of_iterations):
        # observations are for rule based agents
        # processed_observations are for conv net based agents
        observations = env.reset()
        game_states = env.game_states
        actions_1, actions_1_dict, actions_1_probs, proc_obs1, reward1 = agent(observations[0],
                                                                               configuration, game_states[0])
        actions_2, actions_2_dict, actions_2_probs, proc_obs2, reward2 = agent(observations[1],
                                                                               configuration, game_states[1])

        for step in range(1, configuration.episodeSteps):
            dones, observations = env.step((actions_1, actions_2))
            game_states = env.game_states
            actions_1, actions_1_dict, actions_1_probs, proc_obs1, reward1 = agent(observations[0],
                                                                                   configuration, game_states[0])
            actions_2, actions_2_dict, actions_2_probs, proc_obs2, reward2 = agent(observations[1],
                                                                                   configuration, game_states[1])
            if any(dones):
                break


if __name__ == '__main__':
    number_of_games = 10
    policy_agent = agents.get_processing_agent("half_imitator_six_actions")
    show_gym(number_of_games, policy_agent)

    print("Test a submission style agent.")

    policy_agent = agents.get_agent("half_imitator", is_gym=False)
    environment = kaggle.make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2}, debug=True)
    steps = environment.run([policy_agent, policy_agent])

    print("Done.")
