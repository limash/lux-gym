import time
import pickle
import numpy as np
import tensorflow as tf

from lux_ai import models, tools
from lux_gym.envs.lux.action_vectors import meaning_vector, actions_number
import lux_gym.envs.tools as env_tools


def get_policy():
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    actions_shape = actions_number
    units_model = models.actor_critic_base()
    cts_model = models.actor_critic_base()
    dummy_input = tf.ones(feature_maps_shape, dtype=tf.float32)
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    cts_model(dummy_input)
    units_model(dummy_input)
    try:
        with open('data/city_tiles/data.pickle', 'rb') as file:
            cts_data = pickle.load(file)
        with open('data/units/data.pickle', 'rb') as file:
            units_data = pickle.load(file)
        cts_model.set_weights(cts_data['weights'])
        units_model.set_weights(units_data['weights'])
    except FileNotFoundError:
        pass

    @tf.function(experimental_relax_shapes=True)
    def predict_cts(obs, mask):
        return cts_model((obs, mask))

    @tf.function(experimental_relax_shapes=True)
    def predict_units(obs, mask):
        return units_model((obs, mask))

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

        width, height = current_game_state.map.width, current_game_state.map.height
        shift = int((32 - width) / 2)

        # city tiles
        if proc_observations["city_tiles"]:
            t1 = time.perf_counter()
            cts_obs = np.stack(list(proc_observations["city_tiles"].values()), axis=0)
            cts_masks = np.tile(citytile_action_mask, (cts_obs.shape[0], 1))
            cts_obs = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), cts_obs)
            cts_masks = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), cts_masks)
            acts, vals = predict_cts(cts_obs, cts_masks)
            acts = tf.nn.softmax(tf.math.log(acts) * 2)  # sharpen distribution
            t2 = time.perf_counter()
            print(f"2. City Tiles prediction: {t2 - t1:0.4f} seconds")
            t1 = time.perf_counter()
            for i, key in enumerate(proc_observations["city_tiles"].keys()):
                _, x, y = key.split("_")
                x, y = int(y) - shift, int(x) - shift
                citytiles_actions_probs_dict[key] = acts[i, :].numpy()
                max_arg = tf.squeeze(tf.random.categorical(tf.math.log(acts[i:i+1, :]), 1))
                action_one_hot = tf.one_hot(max_arg, actions_shape)
                citytiles_actions_dict[key] = action_one_hot.numpy()
                # deserialization
                meaning = meaning_vector[max_arg.numpy()]
                if meaning[0] == "bw":
                    action_string = f"{meaning[0]} {x} {y}"
                elif meaning[0] == "bc":
                    action_string = f"bw {x} {y}"  # do not build cart
                elif meaning[0] == "r":
                    action_string = f"{meaning[0]} {x} {y}"
                elif meaning[0] == "idle":
                    action_string = None
                else:
                    raise ValueError
                if action_string is not None:
                    actions.append(action_string)
            t2 = time.perf_counter()
            print(f"2. City tiles deserialization: {t2 - t1:0.4f} seconds")

        # workers
        if proc_observations["workers"]:
            t1 = time.perf_counter()
            workers_obs = np.stack(list(proc_observations["workers"].values()), axis=0)
            workers_masks = np.tile(worker_action_mask, (workers_obs.shape[0], 1))
            workers_obs = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), workers_obs)
            workers_masks = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), workers_masks)
            acts, vals = predict_units(workers_obs, workers_masks)
            acts = tf.nn.softmax(tf.math.log(acts) * 2)  # sharpen distribution
            t2 = time.perf_counter()
            print(f"2. Workers prediction: {t2 - t1:0.4f} seconds")
            t1 = time.perf_counter()
            for i, key in enumerate(proc_observations["workers"].keys()):
                workers_actions_probs_dict[key] = acts[i, :].numpy()
                max_arg = tf.squeeze(tf.random.categorical(tf.math.log(acts[i:i+1]), 1))
                action_one_hot = tf.one_hot(max_arg, actions_shape)
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
            t2 = time.perf_counter()
            print(f"2. Workers deserialization: {t2 - t1:0.4f} seconds")

        return actions, actions_dict, actions_probs_dict, proc_observations

    return policy
