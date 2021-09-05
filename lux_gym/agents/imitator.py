import pickle
import tensorflow as tf

from lux_ai import models, tools
from lux_gym.envs.lux.action_vectors import action_vector, meaning_vector
from lux_gym.envs.lux.action_vectors import worker_action_mask, cart_action_mask, citytile_action_mask
import lux_gym.envs.tools as env_tools


def get_policy():
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    actions_shape = len(action_vector)
    model = models.get_actor_critic(feature_maps_shape, actions_shape)
    with open('data/data.pickle', 'rb') as file:
        init_data = pickle.load(file)
    model.set_weights(init_data['weights'])

    def find_unit(player, unit_id, direction):
        dest_id = 'unknown'
        return dest_id

    @tf.function
    def predict(obs, mask):
        return model((obs, mask))

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

        proc_observations = env_tools.get_separate_outputs(observation, current_game_state)
        observations = tf.nest.map_structure(lambda z: tf.expand_dims(z, axis=0), proc_observations)
        observations = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), observations)

        player = current_game_state.players[observation.player]
        width, height = current_game_state.map.width, current_game_state.map.height
        shift = int((32 - width) / 2)

        # workers
        worker_action_mask_exp = tf.expand_dims(worker_action_mask, axis=0)
        worker_action_mask_exp = tf.cast(worker_action_mask_exp, dtype=tf.float32)
        workers = observations["workers"]
        for key, obs in workers.items():
            action, value = predict(obs, worker_action_mask_exp)
            action_masked = action * worker_action_mask_exp
            workers_actions_probs_dict[key] = action_masked[0].numpy()
            max_arg = tf.argmax(action_masked, axis=-1)
            action_one_hot = tf.one_hot(max_arg, actions_shape)
            workers_actions_dict[key] = action_one_hot[0].numpy()

            # deserialization
            meaning = meaning_vector[max_arg[0].numpy()]
            if meaning[0] == "m":
                action_string = f"{meaning[0]} {key} {meaning[1]}"
            elif meaning[0] == "p":
                action_string = f"{meaning[0]} {key}"
            elif meaning[0] == "t":
                direction = meaning[1]
                dest_id = find_unit(player, key, direction)
                action_string = f"{meaning[0]} {key} {dest_id} {meaning[2]} {100}"
            elif meaning[0] == "bcity":
                action_string = f"{meaning[0]} {key}"
            else:
                raise ValueError
            actions.append(action_string)

        # city tiles
        citytile_action_mask_exp = tf.expand_dims(citytile_action_mask, axis=0)
        citytile_action_mask_exp = tf.cast(citytile_action_mask_exp, dtype=tf.float32)
        city_tiles = observations["city_tiles"]
        for key, obs in city_tiles.items():
            _, x, y = key.split("_")
            x, y = int(y) - shift, int(x) - shift
            action, value = predict(obs, citytile_action_mask_exp)
            action_masked = action * citytile_action_mask_exp
            citytiles_actions_probs_dict[key] = action_masked[0].numpy()
            max_arg = tf.argmax(action_masked, axis=-1)
            action_one_hot = tf.one_hot(max_arg, actions_shape)
            citytiles_actions_dict[key] = action_one_hot[0].numpy()

            # deserialization
            meaning = meaning_vector[max_arg[0].numpy()]
            if meaning[0] == "bw":
                action_string = f"{meaning[0]} {x} {y}"
            elif meaning[0] == "bc":
                action_string = f"{meaning[0]} {x} {y}"
            elif meaning[0] == "r":
                action_string = f"{meaning[0]} {x} {y}"
            elif meaning[0] == "idle":
                action_string = None
            else:
                raise ValueError
            if action_string is not None:
                actions.append(action_string)

        return actions, actions_dict, actions_probs_dict, proc_observations

    return policy
