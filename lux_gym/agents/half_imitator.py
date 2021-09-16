import pickle
import tensorflow as tf

from lux_ai import models, tools
from lux_gym.envs.lux.action_vectors import action_vector, meaning_vector
from lux_gym.envs.lux.action_vectors import worker_action_mask, cart_action_mask, citytile_action_mask
import lux_gym.envs.tools as env_tools


def get_policy():
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    actions_shape = len(action_vector)
    model = models.actor_critic_custom()
    dummy_input = (tf.ones(feature_maps_shape, dtype=tf.float32),
                   tf.convert_to_tensor(worker_action_mask, dtype=tf.float32))
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    model(dummy_input)
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
        model.set_weights(init_data['weights'])
    except FileNotFoundError:
        pass

    # def find_unit(player, unit_id, direction):
    #     dest_id = 'unknown'
    #     return dest_id

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

        # if current_game_state.turn == 150:
        #     print("Turn 150")

        proc_observations = env_tools.get_separate_outputs(observation, current_game_state)
        observations = tf.nest.map_structure(lambda z: tf.expand_dims(z, axis=0), proc_observations)
        observations = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), observations)

        player = current_game_state.players[observation.player]
        width, height = current_game_state.map.width, current_game_state.map.height
        shift = int((32 - width) / 2)

        # city tiles
        # max number of units available
        units_cap = sum([len(x.citytiles) for x in player.cities.values()])
        # current number of units
        units = len(player.units)

        for city in player.cities.values():
            created_worker = (units >= units_cap)
            for city_tile in city.citytiles[::-1]:
                if city_tile.can_act():
                    if created_worker:
                        # let's do research
                        action, report = city_tile.research_report(shift)
                        actions.append(action)
                        citytiles_actions_dict[report[0]] = report[1]
                        citytiles_actions_probs_dict[report[0]] = report[1]
                    else:
                        # let's create one more unit in the last created city tile if we can
                        action, report = city_tile.build_worker_report(shift)
                        actions.append(action)
                        citytiles_actions_dict[report[0]] = report[1]
                        citytiles_actions_probs_dict[report[0]] = report[1]
                        created_worker = True

        # workers
        worker_action_mask_exp = tf.expand_dims(worker_action_mask, axis=0)
        worker_action_mask_exp = tf.cast(worker_action_mask_exp, dtype=tf.float32)
        workers = observations["workers"]
        for key, obs in workers.items():
            action, value = predict(obs, worker_action_mask_exp)

            # action_v = action.numpy()
            # action_masked = action * worker_action_mask_exp
            # worker_action = action[0, 4:23]
            # worker_action_v = worker_action.numpy()
            # probs_logs = tf.math.log(worker_action)
            # adjusted_logs = probs_logs * 2
            # adjusted_probs = tf.nn.softmax(adjusted_logs)
            # adjusted_probs_v = adjusted_probs.numpy()
            # action_probs = tf.nn.softmax(tf.math.log(action) * 2)
            # action_probs_v = action_probs.numpy()

            workers_actions_probs_dict[key] = action[0].numpy()
            # max_arg = tf.argmax(action_masked, axis=-1)[0]
            max_arg = tf.squeeze(tf.random.categorical(tf.math.log(action), 1))
            action_one_hot = tf.one_hot(max_arg, actions_shape)
            workers_actions_dict[key] = action_one_hot[0].numpy()

            # deserialization
            meaning = meaning_vector[max_arg.numpy()]
            if meaning[0] == "m":
                action_string = f"{meaning[0]} {key} {meaning[1]}"
            elif meaning[0] == "p":
                action_string = f"{meaning[0]} {key}"
            elif meaning[0] == "t":
                action_string = f"m {key} c"
                # raise NotImplementedError
                # direction = meaning[1]
                # dest_id = find_unit(player, key, direction)
                # action_string = f"{meaning[0]} {key} {dest_id} {meaning[2]} {100}"
            elif meaning[0] == "bcity":
                action_string = f"{meaning[0]} {key}"
            else:
                raise ValueError
            actions.append(action_string)

        # carts
        # cart_action_mask_exp = tf.expand_dims(cart_action_mask, axis=0)
        # cart_action_mask_exp = tf.cast(cart_action_mask_exp, dtype=tf.float32)
        carts = observations["carts"]
        for key, obs in carts.items():
            raise NotImplementedError
            # action, value = predict(obs, cart_action_mask_exp)

        return actions, actions_dict, actions_probs_dict, proc_observations

    return policy
