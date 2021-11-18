from kaggle_environments import make

env = make("lux_ai_2021", configuration={"loglevel": 4}, debug=True)
steps = env.run(["agent_compare.py", "main.py"])
print("done")
