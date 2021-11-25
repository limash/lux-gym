import os
import time
import copy
import pickle
import itertools

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import tools
import hparams
import effnetv2_configs
import utils
from effnetv2_model import round_filters, round_repeats, Stem
from effnetv2_model import MBConvBlock, FusedMBConvBlock
from action_vectors import actions_number
from game import Game

COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200


class EfficientModel(keras.Model):
    def __init__(self, actions_n, **kwargs):
        model_name = 'efficientnetv2-s'
        super().__init__(name=model_name, **kwargs)

        cfg = copy.deepcopy(hparams.base_config)
        if model_name:
            cfg.override(effnetv2_configs.get_model_config(model_name))
        self.cfg = cfg
        self._mconfig = cfg.model

        self._stem = Stem(self._mconfig, self._mconfig.blocks_args[0].input_filters)

        self._blocks = []
        block_id = itertools.count(0)
        block_name = lambda: 'blocks_%d' % next(block_id)
        for block_args in self._mconfig.blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(block_args.input_filters, self._mconfig)
            output_filters = round_filters(block_args.output_filters, self._mconfig)
            repeats = round_repeats(block_args.num_repeat,
                                    self._mconfig.depth_coefficient)
            block_args.update(
                dict(
                    input_filters=input_filters,
                    output_filters=output_filters,
                    num_repeat=repeats))

            # The first block needs to take care of stride and filter size increase.
            conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}[block_args.conv_type]
            self._blocks.append(
                conv_block(block_args, self._mconfig, name=block_name()))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                # pylint: disable=protected-access
                block_args.input_filters = block_args.output_filters
                block_args.strides = 1
                # pylint: enable=protected-access
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    conv_block(block_args, self._mconfig, name=block_name()))

        initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
        activation = utils.get_act_fn(self._mconfig.act_fn)

        self._depthwise = keras.layers.DepthwiseConv2D(13)
        self._flatten = keras.layers.Flatten()

        self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
        self._workers_probs1 = keras.layers.Dense(actions_n, activation="softmax",
                                                  kernel_initializer=initializer_random)
        self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                            activation=keras.activations.tanh)

    def call(self, inputs, training=False, mask=None):
        outputs = self._stem(inputs, training)
        for idx, block in enumerate(self._blocks):
            survival_prob = self._mconfig.survival_prob
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
            # survival_prob = 1.0
            outputs = block(outputs, training=training, survival_prob=survival_prob)

        x = outputs

        shape_x = tf.shape(x)
        y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
        y = tf.reduce_mean(y, axis=1)

        z1 = (x * inputs[:, :, :, :1])
        shape_z = tf.shape(z1)
        z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
        z1 = tf.reduce_sum(z1, axis=1)
        z2 = self._depthwise(x)
        z2 = self._flatten(z2)
        z = tf.concat([z1, z2], axis=1)

        w = self._workers_probs0(z)
        w = self._workers_probs1(w)
        probs = w

        baseline = self._baseline(tf.concat([y, z], axis=1))

        return probs, baseline

    def get_config(self):
        pass


def get_policy():
    feature_maps_shape = (13, 13, 66)
    model = EfficientModel(actions_number)
    dummy_input = tf.ones(feature_maps_shape, dtype=tf.float32)
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
        tools.units_actions_dict.clear()
    else:
        game_state._update(observation["updates"])

    actions, _, _, _ = current_policy(game_state, observation)
    return actions
