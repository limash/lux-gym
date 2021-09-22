import time
import pickle
import numpy as np
import tensorflow as tf

from lux_ai import models, tools
from lux_gym.envs.lux.action_vectors import meaning_vector, actions_number
import lux_gym.envs.tools as env_tools


def get_policy():
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    model = models.actor_critic_base(actions_number)
    dummy_input = tf.ones(feature_maps_shape, dtype=tf.float32)
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    model(dummy_input)
    try:
        with open('data/units/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
        model.set_weights(init_data['weights'])
    except FileNotFoundError:
        pass

    @tf.function(experimental_relax_shapes=True)
    def predict(obs):
        return model(obs)

    def policy(current_game_state, observation):
        actions = []
        workers_actions_probs_dict = {}
        workers_actions_dict = {}
        citytiles_actions_probs_dict = {}
        citytiles_actions_dict = {}
        actions_probs_dict = {"workers": workers_actions_probs_dict,
                              "carts": {},
                              "city_tiles": citytiles_actions_probs_dict}
        actions_dict = {"workers": workers_actions_dict,
                        "carts": {},
                        "city_tiles": citytiles_actions_dict}

        print(f"Step: {observation['step']}; Player: {observation['player']}")
        t1 = time.perf_counter()
        proc_observations = env_tools.get_separate_outputs(observation, current_game_state)
        t2 = time.perf_counter()
        print(f"1. Observations processing: {t2 - t1:0.4f} seconds")

        player = current_game_state.players[observation.player]

        unit_count = len(player.units)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                if city_tile.can_act():
                    if unit_count < player.city_tile_count:
                        actions.append(city_tile.build_worker())
                        unit_count += 1
                    elif not player.researched_uranium():
                        actions.append(city_tile.research())
                        player.research_points += 1

        # workers
        if proc_observations["workers"]:
            t1 = time.perf_counter()
            workers_obs = np.stack(list(proc_observations["workers"].values()), axis=0)
            workers_obs = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), workers_obs)
            acts, vals = predict(workers_obs)
            # acts = tf.nn.softmax(tf.math.log(acts) * 2)  # sharpen distribution
            t2 = time.perf_counter()
            print(f"2. Workers prediction: {t2 - t1:0.4f} seconds")
            for i, key in enumerate(proc_observations["workers"].keys()):
                workers_actions_probs_dict[key] = acts[i, :].numpy()
                max_arg = tf.squeeze(tf.random.categorical(tf.math.log(acts[i:i+1]), 1))
                action_one_hot = tf.one_hot(max_arg, actions_number)
                workers_actions_dict[key] = action_one_hot.numpy()
                # deserialization
                meaning = meaning_vector[max_arg.numpy()]
                if meaning[0] == "m":
                    action_string = f"{meaning[0]} {key} {meaning[1]}"
                elif meaning[0] == "p":
                    action_string = f"{meaning[0]} {key}"
                elif meaning[0] == "t":
                    action_string = f"m {key} c"  # move center instead
                elif meaning[0] == "bcity":
                    action_string = f"{meaning[0]} {key}"
                else:
                    raise ValueError
                actions.append(action_string)

        return actions, actions_dict, actions_probs_dict, proc_observations

    return policy
