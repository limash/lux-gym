import os
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import tools
from action_vectors import meaning_vector, actions_number
from game import Game  # , Missions
from actions import make_city_actions

# missions = Missions()


def squeeze_transform(obs_base, acts_rews):
    actions_probs, total_rewards = acts_rews
    features = obs_base
    # features_v = features.numpy()

    features_padded = tf.pad(features, tf.constant([[6, 6], [6, 6], [0, 0]]), mode="CONSTANT")
    # features_padded_v = features_padded.numpy()
    units_layers = features_padded[:, :, :1]
    units_coords = tf.cast(tf.where(units_layers), dtype=tf.int32)
    min_x = units_coords[:, 0] - 6
    max_x = units_coords[:, 0] + 6
    min_y = units_coords[:, 1] - 6
    max_y = units_coords[:, 1] + 6
    piece = features_padded[min_x[0]: max_x[0] + 1, min_y[0]: max_y[0] + 1, :]
    # piece_v = piece.numpy()

    features_padded_glob = tf.pad(features,
                                  tf.constant([[32, 32], [32, 32], [0, 0]]),
                                  mode="CONSTANT")
    # features_padded_glob_v = features_padded_glob.numpy()
    units_layers_glob = features_padded_glob[:, :, :1]
    units_coords_glob = tf.cast(tf.where(units_layers_glob), dtype=tf.int32)
    min_x_glob = units_coords_glob[:, 0] - 32
    max_x_glob = units_coords_glob[:, 0] + 32
    min_y_glob = units_coords_glob[:, 1] - 32
    max_y_glob = units_coords_glob[:, 1] + 32

    piece_glob1 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 6:9]
    piece_glob2 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 15:17]
    piece_glob3 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 24:27]
    piece_glob4 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 45:49]
    piece_glob5 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 60:]
    piece_glob = tf.concat([piece_glob1, piece_glob2, piece_glob3, piece_glob4, piece_glob5], axis=-1)
    # 17, 4; 5, 5
    pooled_piece_glob = tf.squeeze(tf.nn.avg_pool(tf.expand_dims(piece_glob, axis=0), 5, 5, padding="VALID"))
    # piece_glob_v = piece_glob.numpy()
    # pooled_piece_glob_v = pooled_piece_glob.numpy()

    piece_filtered = tf.concat([piece[:, :, :1],
                                piece[:, :, 4:10],
                                piece[:, :, 15:50],
                                piece[:, :, 60:],
                                ], axis=-1)
    # piece_filtered_v = piece_filtered.numpy()
    observations = tf.concat([piece_filtered, pooled_piece_glob], axis=-1)
    return observations, (actions_probs, total_rewards)


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
    def __init__(self, actions_n, **kwargs):
        super().__init__(**kwargs)

        filters = 64
        layers = 12

        initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
        activation = keras.activations.relu

        self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
        self._norm = keras.layers.BatchNormalization()
        self._activation = keras.layers.ReLU()
        self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

        self._depthwise = keras.layers.DepthwiseConv2D(13)
        self._flatten = keras.layers.Flatten()

        # self._city_tiles_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        # self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax",
        #                                              kernel_initializer=initializer_random)
        self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        self._workers_probs1 = keras.layers.Dense(actions_n, activation="softmax",
                                                  kernel_initializer=initializer_random)
        # self._carts_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        # self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

        self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                            activation=keras.activations.tanh)

    def call(self, inputs, training=False, mask=None):
        features = inputs
        # features = tf.concat([inputs[:, :, :, :36], inputs[:, :, :, 37:38], inputs[:, :, :, 39:]], axis=-1)
        # features = tf.concat([inputs[:, :, :, :1],
        #                       inputs[:, :, :, 4:10],
        #                       inputs[:, :, :, 15:42],
        #                       inputs[:, :, :, 43:44],
        #                       inputs[:, :, :, 45:],
        #                       ], axis=-1)

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


def get_policy():
    feature_maps_shape = (32, 32, 61)
    model = ResidualModel(actions_number)
    dummy_input = tf.ones(feature_maps_shape, dtype=tf.float32)
    dummy_input, (_, _) = squeeze_transform(dummy_input, (None, None))
    dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    model(dummy_input)
    path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
    with open(f'{path}/data.pickle', 'rb') as file:
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

    def call_func(obj, method, args=[]):
        return getattr(obj, method)(*args)

    unit_actions = [('move', 'n'), ('move', 'e'), ('move', 's'), ('move', 'w'), ('move', 'c'), ('build_city',)]

    def get_action(game_state, plc, unit, dest):
        for label in np.argsort(plc)[::-1]:
            act = unit_actions[label]
            pos = unit.pos.translate(act[-1], 1) or unit.pos
            if pos not in dest or in_city(game_state, pos):
                return call_func(unit, *act), pos

        return unit.move('c'), unit.pos

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
        for key, value in proc_observations["workers"].items():
            value = tf.cast(value, dtype=tf.float32)
            obs_squeezed, (_, _) = squeeze_transform(value, (None, None))
            proc_observations["workers"][key] = obs_squeezed
        t2 = time.perf_counter()
        print(f"1. Observations processing: {t2 - t1:0.4f} seconds")

        player = current_game_state.players[observation.player]
        n_city_tiles = player.city_tile_count
        unit_count = len(player.units)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                if city_tile.can_act():
                    if unit_count < player.city_tile_count:
                        actions.append(city_tile.build_worker())
                        unit_count += 1
                    elif not player.researched_uranium() and n_city_tiles > 2:
                        actions.append(city_tile.research())
                        player.research_points += 1

        # t1 = time.perf_counter()
        # current_game_state.calculate_features(missions)
        # actions_by_cities = make_city_actions(current_game_state, missions)
        # actions += actions_by_cities
        # t2 = time.perf_counter()
        # print(f"2. City tiles prediction: {t2 - t1:0.4f} seconds")

        # workers
        dest = []
        if proc_observations["workers"]:
            t1 = time.perf_counter()
            workers_obs = np.stack(list(proc_observations["workers"].values()), axis=0)
            workers_obs = tf.nest.map_structure(lambda z: tf.cast(z, dtype=tf.float32), workers_obs)
            acts, vals = predict(workers_obs)
            # acts = tf.nn.softmax(tf.math.log(acts) * 2)  # sharpen distribution
            logs = tf.math.log(acts)
            t2 = time.perf_counter()
            print(f"2. Workers prediction: {t2 - t1:0.4f} seconds")
            for i, key in enumerate(proc_observations["workers"].keys()):
                workers_actions_probs_dict[key] = acts[i, :].numpy()
                max_arg = tf.squeeze(tf.random.categorical(tf.math.log(acts[i:i+1]), 1))
                action_one_hot = tf.one_hot(max_arg, actions_number)
                workers_actions_dict[key] = action_one_hot.numpy()
                # filter bad actions
                unit = player.units_by_id[key]
                pol = logs[i, :].numpy()
                action, pos = get_action(current_game_state, pol, unit, dest)
                actions.append(action)
                dest.append(pos)
                # deserialization
                # meaning = meaning_vector[max_arg.numpy()]
                # if meaning[0] == "m":
                #     action_string = f"{meaning[0]} {key} {meaning[1]}"
                # elif meaning[0] == "p":
                #     action_string = f"{meaning[0]} {key}"
                # elif meaning[0] == "t":
                #     action_string = f"m {key} c"  # move center instead
                # elif meaning[0] == "bcity":
                #     action_string = f"{meaning[0]} {key}"
                # else:
                #     raise ValueError
                # actions.append(action_string)

        return actions, actions_dict, actions_probs_dict, proc_observations
    return policy


game_state = None
current_policy = get_policy()


def agent(observation, configuration):
    global game_state

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state.player_id = observation.player
        game_state._update(observation["updates"][2:])
        # game_state.id = observation.player
        game_state.fix_iteration_order()
    else:
        game_state._update(observation["updates"])

    actions, _, _, _ = current_policy(game_state, observation)
    return actions
