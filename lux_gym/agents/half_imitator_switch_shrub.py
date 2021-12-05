import os
import time
import pickle
import numpy as np
import tensorflow as tf

import lux_gym.envs.tools as env_tools
from lux_ai import models, tools
# from lux_gym.envs.lux.action_vectors import actions_number
from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors, worker_action_vector, dir_action_vector

COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200


def get_policy(init_data=None):
    feature_maps_shape = tools.get_feature_maps_shape('lux_gym:lux-v0')
    model = models.actor_critic_residual_switch_shrub()
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

    unit_actions = [('move', 'n'), ('move', 'e'), ('move', 's'), ('move', 'w'), ('build_city',), ('move', 'c')]
    directions = ['n', 'e', 's', 'w']
    resources = ['wood', 'coal', 'uranium']

    def policy(current_game_state, observation):
        # global missions

        actions = []

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
            t1 = time.perf_counter()
            workers_obs = np.stack(list(proc_observations["workers"].values()), axis=0)
            ns, es, ss, ws, bs, idles, ts, t_probs, t_res = predict(workers_obs)
            ons = tf.stack([ns[:, 1], es[:, 1], ss[:, 1], ws[:, 1], bs[:, 1], idles[:, 1], ts[:, 1]], axis=1)
            t2 = time.perf_counter()
            print(f"2. Workers prediction: {t2 - t1:0.4f} seconds")
            for i, key in enumerate(proc_observations["workers"].keys()):
                unit = player.units_by_id[key]
                movement_probs = ons[i, :4]
                mov_prob = tf.reduce_max(movement_probs)
                if mov_prob > 0.9:
                    probs = movement_probs
                else:
                    probs = ons[i, :]
                label = tf.argmax(probs)
                if label != 6:  # 6 is transfer
                    act = unit_actions[label]
                    pos = unit.pos.translate(act[-1], 1) or unit.pos
                    if pos not in dest or in_city(current_game_state, pos):
                        action = call_func(unit, *act)
                    else:
                        action = unit.move('c')
                else:
                    dir_label = tf.argmax(t_probs[i, :])
                    direction = directions[dir_label]
                    trans_pos = unit.pos.translate(direction, 1) or unit.pos
                    try:
                        dest_unit = current_game_state.map.get_cell_by_pos(trans_pos).unit
                    except IndexError:
                        dest_unit = None
                    if dest_unit is not None and dest_unit.team == current_game_state.player_id:
                        resource_label = tf.argmax(t_res[i, :])
                        resourceType = resources[resource_label]
                        action = unit.transfer(dest_unit.id, resourceType, 2000)
                    else:
                        act = unit_actions[4]  # idle
                        action = call_func(unit, *act)
                    pos = unit.pos

                actions.append(action)
                dest.append(pos)

        return actions, None, None, proc_observations

    return policy
