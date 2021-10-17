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


def actor_critic_efficient(actions_shape):
    import copy
    import itertools

    import tensorflow as tf
    import tensorflow.keras as keras

    # import lux_ai.effnetv2_model as eff_model
    import hparams
    import effnetv2_configs
    import utils
    from effnetv2_model import round_filters, round_repeats, Stem, MBConvBlock, FusedMBConvBlock

    class EfficientModel(keras.Model):
        def __init__(self, actions_number, **kwargs):
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
            self._workers_probs1 = keras.layers.Dense(actions_number, activation="softmax",
                                                      kernel_initializer=initializer_random)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            outputs = self._stem(inputs, training)
            for idx, block in enumerate(self._blocks):
                # survival_prob = self._mconfig.survival_prob
                # if survival_prob:
                #     drop_rate = 1.0 - survival_prob
                #     survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
                survival_prob = 1.0
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

    model = EfficientModel(actions_shape)
    # model = eff_model.get_model("efficientnetv2-s", weights=None)
    return model


def get_policy():
    feature_maps_shape = (13, 13, 64)
    model = actor_critic_efficient(actions_number)
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
