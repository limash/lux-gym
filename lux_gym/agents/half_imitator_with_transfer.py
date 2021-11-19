import os
# import time
import pickle
import numpy as np
import tensorflow as tf

import lux_gym.envs.tools as env_tools
from lux_ai import models, tools
from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors, worker_action_vector
from lux_gym.envs.lux.action_vectors_new import dir_action_vector, res_action_vector

COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200


def get_policy(init_data=None):
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    model = models.actor_critic_residual_with_transfer()
    dummy_input = tf.ones(feature_maps_shape, dtype=tf.float32)
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    model(dummy_input)
    if init_data is not None:
        model.set_weights(init_data['weights'])
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.split(os.path.split(dir_path)[0])[0]
        try:
            with open(dir_path + '/data/units/data.pickle', 'rb') as file:
                init_data = pickle.load(file)
            model.set_weights(init_data['weights'])
        except FileNotFoundError:
            pass

    @tf.function(experimental_relax_shapes=True)
    def predict(obs):
        return model(obs)

    def in_city(game_state, pos):
        try:
            city = game_state.map.get_cell_by_pos(pos).citytile
            return city is not None and city.team == game_state.player_id
        except:
            return False

    def call_func(obj, method, args=None):
        if args is None:
            args = []
        return getattr(obj, method)(*args)

    unit_actions = [('move', 'n'), ('move', 'e'), ('move', 's'), ('move', 'w'), ('move', 'c'), ('build_city',)]
    directions = ['n', 'e', 's', 'w']
    resources = ['wood', 'coal', 'uranium']

    def get_action(game_state, action_logs, trans_dirs, resource_types, unit, dest):
        # multiplier = 1
        # for _ in range(4):
        for label in np.argsort(tf.squeeze(action_logs))[::-1]:
            # label = tf.squeeze(tf.random.categorical(action_logs / multiplier, 1))
            if label != 6:  # 6 is transfer
                act = unit_actions[label]
                pos = unit.pos.translate(act[-1], 1) or unit.pos
            else:
                dir_label = np.argmax(trans_dirs)
                direction = directions[dir_label]
                trans_pos = unit.pos.translate(direction, 1) or unit.pos
                try:
                    dest_unit = game_state.map.get_cell_by_pos(trans_pos).unit
                except IndexError:
                    dest_unit = None
                if dest_unit is not None and dest_unit.team == game_state.player_id:
                    resource_label = np.argmax(resource_types)
                    resourceType = resources[resource_label]
                    act = ('transfer', dest_unit.id, resourceType, 2000)
                else:
                    act = unit_actions[4]  # idle
                    label = 4  # idle
                pos = unit.pos
            if pos not in dest or in_city(game_state, pos):
                return label, call_func(unit, *act), pos
            # multiplier *= 2

        return 4, unit.move('c'), unit.pos  # 4 is idle

    def policy(current_game_state, observation):
        # global missions

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

        # print(f"Step: {observation['step']}; Player: {observation['player']}")
        # t1 = time.perf_counter()
        proc_observations = env_tools.get_separate_outputs(observation, current_game_state)
        # t2 = time.perf_counter()
        # print(f"1. Observations processing: {t2 - t1:0.4f} seconds")

        total_resources = 0
        player = current_game_state.players[observation.player]
        player_research_points = player.research_points
        width, height = current_game_state.map.width, current_game_state.map.height
        for y in range(height):
            for x in range(width):
                cell = current_game_state.map.get_cell(x, y)
                if cell.has_resource():
                    resource = cell.resource
                    if resource.type == "wood":
                        total_resources += 1
                    elif resource.type == "coal":
                        if player_research_points >= COAL_RESEARCH_POINTS - 10:
                            total_resources += 1
                    elif resource.type == "uranium":
                        if player_research_points >= URAN_RESEARCH_POINTS - 10:
                            total_resources += 1
                    else:
                        raise ValueError

        n_city_tiles = player.city_tile_count
        unit_count = len(player.units)
        for city in reversed(player.cities.values()):
            for city_tile in reversed(city.citytiles):
                if city_tile.can_act():
                    if unit_count < player.city_tile_count and unit_count < total_resources:
                        actions.append(city_tile.build_worker())
                        unit_count += 1
                    elif not player.researched_uranium() and n_city_tiles > 2:
                        actions.append(city_tile.research())
                        player.research_points += 1

        # workers
        dest = []
        directions_dict = {0: "n", 1: "e", 2: "s", 3: "w"}
        resources_dict = {0: "wood", 1: "coal", 2: "uranium"}
        if proc_observations["workers"]:
            # t1 = time.perf_counter()
            workers_obs = np.stack(list(proc_observations["workers"].values()), axis=0)
            # workers_obs = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), workers_obs)
            action_type_probs, transfer_direction_probs, transfer_resource_probs, baseline = predict(workers_obs)

            # action_type_probs = tf.nn.softmax(tf.random.uniform([1, 7]))
            # transfer_direction_probs = tf.nn.softmax(tf.random.uniform([1, 4]))
            # transfer_resource_probs = tf.nn.softmax(tf.random.uniform([1, 3]))

            action_logs = tf.math.log(tf.clip_by_value(action_type_probs, 1.e-16, 1.))
            # acts = tf.nn.softmax(tf.math.log(acts) * 2)  # sharpen distribution
            # t2 = time.perf_counter()
            # print(f"2. Workers prediction: {t2 - t1:0.4f} seconds")
            for i, key in enumerate(proc_observations["workers"].keys()):
                # filter bad actions and make actions according to probs
                unit = player.units_by_id[key]
                current_arg, action, pos = get_action(current_game_state, action_logs[i:i + 1, :].numpy(),
                                                      transfer_direction_probs[i:i + 1, :].numpy(),
                                                      transfer_resource_probs[i:i + 1, :].numpy(),
                                                      unit, dest)
                actions.append(action)
                dest.append(pos)

                # prepare action dictionaries to save, if necessary
                action_vectors = empty_worker_action_vectors.copy()
                current_arg = int(current_arg)
                if current_arg in {0, 1, 2, 3}:
                    action_type = "m"
                    dir_type = directions_dict[current_arg]
                    action_vectors[1] = dir_action_vector[dir_type]
                elif current_arg == 4:
                    action_type = "idle"
                elif current_arg == 5:
                    action_type = "bcity"
                elif current_arg == 6:
                    action_type = "t"
                    dir_type = directions_dict[int(np.argmax(transfer_direction_probs[i, :].numpy()))]
                    res_type = resources_dict[int(np.argmax(transfer_resource_probs[i, :].numpy()))]
                    action_vectors[1] = dir_action_vector[dir_type]
                    action_vectors[2] = res_action_vector[res_type]
                else:
                    raise ValueError
                action_vectors[0] = worker_action_vector[action_type]

                current_act_probs = action_type_probs[i, :].numpy()
                action_vectors_probs = empty_worker_action_vectors.copy()
                action_vectors_probs[0] = np.hstack((np.sum(current_act_probs[:4]),  # move
                                                     current_act_probs[6:],  # transfer
                                                     current_act_probs[4:6],  # idle, bcity
                                                     )).astype(dtype=np.half)
                # general_logs = tf.math.log(tf.clip_by_value(general_probs, 1.e-16, 1.))
                # action_vectors_probs[0] = tf.nn.softmax(general_logs)
                if action_type == "m":
                    dir_probs = tf.nn.softmax(action_logs[i:i + 1, :4])[0]  # normalize action probs
                    action_vectors_probs[1] = dir_probs.numpy().astype(dtype=np.half)
                elif action_type == "t":
                    action_vectors_probs[1] = transfer_direction_probs[i, :].numpy()
                    action_vectors_probs[2] = transfer_resource_probs[i, :].numpy()
                workers_actions_dict[key] = action_vectors
                workers_actions_probs_dict[key] = action_vectors_probs

        return actions, actions_dict, actions_probs_dict, proc_observations

    return policy
