import os
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import lux_gym.agents.compare_agent_3.tools_compare as tools
from lux_gym.envs.lux.action_vectors import actions_number


COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, initializer, activation, **kwargs):
        super().__init__(**kwargs)

        self._filters = filters
        self._activation = activation
        self._conv = keras.layers.Conv2D(filters, 3, kernel_initializer=initializer, padding="same", use_bias=False)
        self._norm = keras.layers.BatchNormalization()

    def call(self, inputs, training=False, **kwargs):
        x = self._conv(inputs)
        x = self._norm(x, training=training)
        return self._activation(inputs + x)

    def compute_output_shape(self, batch_input_shape):
        batch, x, y, _ = batch_input_shape
        return [batch, x, y, self._filters]


class ResidualModel(keras.Model):
    def __init__(self, actions_number, **kwargs):
        super().__init__(**kwargs)

        filters = 200
        layers = 10

        initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
        activation = keras.activations.relu

        self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
        self._norm = keras.layers.BatchNormalization()
        self._activation = keras.layers.ReLU()
        self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

        # self._depthwise = keras.layers.DepthwiseConv2D(32)
        self._depthwise = keras.layers.DepthwiseConv2D(13)
        self._flatten = keras.layers.Flatten()

        # self._city_tiles_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        # self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax",
        #                                              kernel_initializer=initializer_random)
        self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        self._workers_probs1 = keras.layers.Dense(actions_number, activation="softmax",
                                                  kernel_initializer=initializer_random)
        # self._carts_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        # self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

        self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                            activation=keras.activations.tanh)

    def call(self, inputs, training=False, mask=None):
        features = inputs
        x = features

        x = self._conv(x)
        x = self._norm(x, training=training)
        x = self._activation(x)

        for layer in self._residual_block:
            x = layer(x, training=training)

        shape_x = tf.shape(x)
        y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
        y = tf.reduce_mean(y, axis=1)

        z1 = (x * features[:, :, :, :1])
        shape_z = tf.shape(z1)
        z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
        z1 = tf.reduce_sum(z1, axis=1)
        z2 = self._depthwise(x)
        z2 = self._flatten(z2)
        z = tf.concat([z1, z2], axis=1)

        # t = self._city_tiles_probs0(z)
        # t = self._city_tiles_probs1(t)
        w = self._workers_probs0(z)
        w = self._workers_probs1(w)
        # c = self._carts_probs0(z)
        # c = self._carts_probs1(c)
        # probs = tf.concat([t, w, c], axis=1)
        # probs = probs * actions_mask
        probs = w

        baseline = self._baseline(tf.concat([y, z], axis=1))

        return probs, baseline

    def get_config(self):
        pass


def get_policy(init_data=None):
    feature_maps_shape = (13, 13, 66)
    model = ResidualModel(actions_number)
    dummy_input = tf.ones(feature_maps_shape, dtype=tf.float32)
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    model(dummy_input)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f'{dir_path}/data_compare.pickle', 'rb') as file:
        init_data = pickle.load(file)
    model.set_weights(init_data['weights'])

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

        print(f"Step: {observation['step']}; Player: {observation['player']}")
        t1 = time.perf_counter()
        proc_observations = tools.get_separate_outputs(observation, current_game_state)
        t2 = time.perf_counter()
        print(f"1. Observations processing: {t2 - t1:0.4f} seconds")

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

        cities_names = list(player.cities.keys())
        # for city in reversed(player.cities.values()):  # not available in python 3.7
        for city_name in reversed(cities_names):
            city = player.cities[city_name]
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
                # current_act_probs = action_probs[i, :].numpy()
                # action_vectors = empty_worker_action_vectors.copy()
                # action_vectors_probs = empty_worker_action_vectors.copy()
                # current_arg = int(current_arg)
                # if current_arg in {0, 1, 2, 3}:
                #     action_type = "m"
                #     dir_type = {0: "n", 1: "e", 2: "s", 3: "w"}[current_arg]
                #     action_vectors[1] = dir_action_vector[dir_type]
                # elif current_arg == 4:
                #     action_type = "idle"
                # elif current_arg == 5:
                #     action_type = "bcity"
                # else:
                #     raise ValueError
                # action_vectors[0] = worker_action_vector[action_type]
                # action_vectors_probs[0] = np.hstack((np.sum(current_act_probs[:4]),  # move
                #                                      np.array([0.]),  # transfer
                #                                      current_act_probs[4:],  # idle, bcity
                #                                      )).astype(dtype=np.half)
                # act_probs = tf.nn.softmax(action_logs[i:i + 1, :4])[0]  # normalize action probs
                # action_vectors_probs[1] = act_probs.numpy().astype(dtype=np.half)
                # # action_one_hot = tf.one_hot(max_arg, actions_number)
                # workers_actions_dict[key] = action_vectors
                # workers_actions_probs_dict[key] = action_vectors_probs

        return actions, actions_dict, actions_probs_dict, proc_observations
    return policy
