import os
# import time
import pickle
import numpy as np
import tensorflow as tf

import lux_gym.envs.tools as env_tools
from lux_ai import models, tools
from lux_gym.envs.lux.action_vectors import actions_number
from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors, worker_action_vector, dir_action_vector

COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200


def get_policy(init_data=None):
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    model = models.actor_critic_efficient_six_actions(actions_number)
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

    def get_action(game_state, action_logs, unit, dest):
        # multiplier = 1
        # for _ in range(4):
        for label in np.argsort(tf.squeeze(action_logs))[::-1]:
            # label = tf.squeeze(tf.random.categorical(action_logs / multiplier, 1))
            act = unit_actions[label]
            pos = unit.pos.translate(act[-1], 1) or unit.pos
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
        if proc_observations["workers"]:
            # t1 = time.perf_counter()
            workers_obs = np.stack(list(proc_observations["workers"].values()), axis=0)
            # workers_obs = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), workers_obs)
            action_probs, vals = predict(workers_obs)
            action_logs = tf.math.log(action_probs)
            # acts = tf.nn.softmax(tf.math.log(acts) * 2)  # sharpen distribution
            # t2 = time.perf_counter()
            # print(f"2. Workers prediction: {t2 - t1:0.4f} seconds")
            for i, key in enumerate(proc_observations["workers"].keys()):
                # filter bad actions and make actions according to probs
                unit = player.units_by_id[key]
                current_arg, action, pos = get_action(current_game_state, action_logs[i:i + 1, :].numpy(), unit, dest)
                actions.append(action)
                dest.append(pos)

                # prepare action dictionaries to save, if necessary
                current_act_probs = action_probs[i, :].numpy()
                action_vectors = empty_worker_action_vectors.copy()
                action_vectors_probs = empty_worker_action_vectors.copy()
                current_arg = int(current_arg)
                if current_arg in {0, 1, 2, 3}:
                    action_type = "m"
                    dir_type = {0: "n", 1: "e", 2: "s", 3: "w"}[current_arg]
                    action_vectors[1] = dir_action_vector[dir_type]
                elif current_arg == 4:
                    action_type = "idle"
                elif current_arg == 5:
                    action_type = "bcity"
                else:
                    raise ValueError
                action_vectors[0] = worker_action_vector[action_type]
                action_vectors_probs[0] = np.hstack((np.sum(current_act_probs[:4]),  # move
                                                     np.array([0.]),  # transfer
                                                     current_act_probs[4:],  # idle, bcity
                                                     )).astype(dtype=np.half)
                act_probs = tf.nn.softmax(action_logs[i:i + 1, :4])[0]  # normalize action probs
                action_vectors_probs[1] = act_probs.numpy().astype(dtype=np.half)
                # action_one_hot = tf.one_hot(max_arg, actions_number)
                workers_actions_dict[key] = action_vectors
                workers_actions_probs_dict[key] = action_vectors_probs

        return actions, actions_dict, actions_probs_dict, proc_observations

    return policy
