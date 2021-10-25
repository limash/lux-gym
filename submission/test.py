from kaggle_environments import make

env = make("lux_ai_2021", configuration={"seed": 217108093, "loglevel": 4}, debug=True)
steps = env.run(["compare_agent.py", "main.py"])
