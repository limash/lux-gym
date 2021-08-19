import gym
# import kaggle_environments as kaggle

import lux_gym.agents.rule_agent as rule_agent


def show_gym(number_of_iterations):
    env = gym.make('lux_gym:lux-v0', debug=True)
    configuration = env.configuration()
    policy = rule_agent.agent
    for i in range(number_of_iterations):
        states = env.reset()
        actions_1 = policy(states[0], configuration)
        actions_2 = policy(states[1], configuration)

        for step in range(configuration.episodeSteps):
            dones, states = env.step((actions_1, actions_2))
            actions_1 = policy(states[0], configuration)
            actions_2 = policy(states[1], configuration)
            if any(dones):
                break


if __name__ == '__main__':

    number_of_games = 10
    show_gym(number_of_games)

    # environment = kaggle.make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2}, debug=True)
    # steps = environment.run([rule_agent.agent, rule_agent.agent])

    print("Done")
