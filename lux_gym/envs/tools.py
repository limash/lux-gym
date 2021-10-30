import math

import numpy as np
import tensorflow as tf

# constants from game rules
MAX_MAP_SIDE = 32
WOOD_FUEL_VALUE = 1
COAL_FUEL_VALUE = 5
URAN_FUEL_VALUE = 20
COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200
MAX_ROAD = 6
WORKERS_CARGO = 100
CART_CARGO = 2000
DAY_LENGTH = 30
NIGHT_LENGTH = 10
MAX_DAYS = 360
CYCLE_LENGTH = DAY_LENGTH + NIGHT_LENGTH
TOTAL_CYCLES = MAX_DAYS / (DAY_LENGTH + NIGHT_LENGTH)
# constants for quasi normalization, some variables can be larger than 1
# resources and fuel
WOOD_BOUND = 500
COAL_BOUND = 500
URAN_BOUND = 500
FUEL_BOUND = 10000
# units and cities
UNITS_BOUND = 50
WORKERS_BOUND = UNITS_BOUND
CARTS_BOUND = UNITS_BOUND
CITY_TILES_BOUND = UNITS_BOUND  # since the units number is limited by the city tiles number
CITY_TILES_IN_CITY_BOUND = 25
# from https://www.kaggle.com/c/lux-ai-2021/discussion/265886
UPKEEP_BOUND = 10 * CITY_TILES_IN_CITY_BOUND + 20 * math.sqrt(CITY_TILES_IN_CITY_BOUND)
UPKEEP_BOUND_PER_TILE = UPKEEP_BOUND / CITY_TILES_IN_CITY_BOUND
CITIES_BOUND = 10

units_actions_dict = {}


def to_binary(d, m=8):
    """
    Args:
        d: is an array of decimal numbers to convert to binary
        m: is a number of positions in a binary number, 8 is enough for up to 256 decimal, 256 is 2^8
    Returns:
        np.ndarray of binary representation of d
    """

    reversed_order = ((d[:, None] & (1 << np.arange(m))) > 0).astype(np.half)
    return np.fliplr(reversed_order)


def get_timing(turn):
    current_cycle = turn // CYCLE_LENGTH
    turns_before_current_cycle = current_cycle * CYCLE_LENGTH
    turns_in_cycle = turn - turns_before_current_cycle

    to_next_day = CYCLE_LENGTH - turns_in_cycle
    if turns_in_cycle < DAY_LENGTH:
        is_night = 0
        to_next_night = DAY_LENGTH - turns_in_cycle
    else:
        is_night = 1
        to_next_night = to_next_day + DAY_LENGTH
    return current_cycle + 1, to_next_day, to_next_night, is_night


def test_get_timing():
    for turn in range(360):
        current_cycle, to_next_day, to_next_night, is_night = get_timing(turn)
        print(f"Current turn: {turn}; current cycle: {current_cycle}; "
              f"to next day: {to_next_day}; to next night: {to_next_night}; is night: {is_night}")


def process(observation, current_game_state):
    """
    Args:
        observation: An observation, which agents get as an input from kaggle environment.
        current_game_state: An object provided by kaggle to simplify game info extraction.
    Returns:
        processed_observations: A prepared observation to save to the buffer.
    """

    global units_actions_dict

    player = current_game_state.players[observation.player]
    opponent = current_game_state.players[(observation.player + 1) % 2]
    width, height = current_game_state.map.width, current_game_state.map.height
    shift = int((MAX_MAP_SIDE - width) / 2)  # to make all feature maps 32x32
    turn = current_game_state.turn

    player_units_coords = {}
    player_city_tiles_coords = {}

    player_research_points = player.research_points
    player_city_tiles_count = player.city_tile_count
    player_cities_count = len(player.cities)
    player_units_count = len(player.units)
    player_workers_count = 0
    player_carts_count = 0
    for unit in player.units:
        if unit.is_worker():
            player_workers_count += 1
        elif unit.is_cart():
            player_carts_count += 1
        else:
            raise ValueError

    opponent_research_points = opponent.research_points
    opponent_city_tiles_count = opponent.city_tile_count
    opponent_cities_count = len(opponent.cities)
    opponent_units_count = len(opponent.units)
    opponent_workers_count = 0
    opponent_carts_count = 0
    for unit in opponent.units:
        if unit.is_worker():
            opponent_workers_count += 1
        elif unit.is_cart():
            opponent_carts_count += 1
        else:
            raise ValueError

    current_cycle, to_next_day, to_next_night, is_night = get_timing(turn)

    # map data, define resources and roads, 0 or 1 for bool, 0 to around 1 for float;
    # layers:
    # 0 - a resource
    # 1 - is wood
    # 2 - wood amount
    # 3 - is coal
    # 4 - coal amount
    # 5 - is uranium
    # 6 - uranium amount
    # 7 - fuel equivalent
    # 8 - if a resource is available for the player, 1 when ready
    # 9 - a road lvl
    # 10 - 19 for coordinates

    number_of_resources_layers = 10
    A1 = np.zeros((number_of_resources_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
    for yy in range(height):
        for xx in range(width):
            cell = current_game_state.map.get_cell(xx, yy)
            x, y = yy + shift, xx + shift
            if cell.has_resource():
                A1[0, x, y] = 1  # a resource at the point
                resource = cell.resource
                if resource.type == "wood":
                    A1[1, x, y] = 1
                    wood_amount = resource.amount
                    A1[2, x, y] = wood_amount / WOOD_BOUND
                    fuel = wood_amount * WOOD_FUEL_VALUE
                    A1[8, x, y] = 1  # wood is always available
                elif resource.type == "coal":
                    A1[3, x, y] = 1
                    coal_amount = resource.amount
                    A1[4, x, y] = coal_amount / COAL_BOUND
                    fuel = coal_amount * COAL_FUEL_VALUE
                    A1[8, x, y] = min(player_research_points / COAL_RESEARCH_POINTS, 1)
                elif resource.type == "uranium":
                    A1[5, x, y] = 1
                    uran_amount = resource.amount
                    A1[6, x, y] = uran_amount / URAN_BOUND
                    fuel = uran_amount * URAN_FUEL_VALUE
                    A1[8, x, y] = min(player_research_points / URAN_RESEARCH_POINTS, 1)
                else:
                    raise ValueError
                A1[7, x, y] = fuel / FUEL_BOUND
            A1[9, x, y] = cell.road / MAX_ROAD
    #         A1[10:15, x, y] = to_binary(np.asarray((x,), dtype=np.uint8), m=5)
    #         A1[15:20, x, y] = to_binary(np.asarray((y,), dtype=np.uint8), m=5)

    # map data, define resources and roads, 0 or 1 for bool, 0 to around 1 for float;
    # layers:
    # 0 - a resource
    # 1 - is available
    # 2 - amount
    # 3 - fuel equivalent
    # 4 - a road lvl
    # 5 - next available resource
    # number_of_resources_layers = 6
    # A1 = np.zeros((number_of_resources_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
    # for yy in range(height):
    #     for xx in range(width):
    #         cell = current_game_state.map.get_cell(xx, yy)
    #         x, y = yy + shift, xx + shift
    #         if cell.has_resource():
    #             A1[0, x, y] = 1  # a resource at the point
    #             resource = cell.resource
    #             fuel = 0
    #             if resource.type == "wood":
    #                 A1[1, x, y] = 1
    #                 wood_amount = resource.amount
    #                 A1[2, x, y] = wood_amount / WOOD_BOUND
    #                 fuel = wood_amount * WOOD_FUEL_VALUE
    #             elif resource.type == "coal":
    #                 if player_research_points >= COAL_RESEARCH_POINTS:
    #                     A1[1, x, y] = 1
    #                     coal_amount = resource.amount
    #                     A1[2, x, y] = coal_amount / COAL_BOUND
    #                     fuel = coal_amount * COAL_FUEL_VALUE
    #                 else:
    #                     A1[5, x, y] = 1
    #             elif resource.type == "uranium":
    #                 if player_research_points >= URAN_RESEARCH_POINTS:
    #                     A1[1, x, y] = 1
    #                     uran_amount = resource.amount
    #                     A1[2, x, y] = uran_amount / URAN_BOUND
    #                     fuel = uran_amount * URAN_FUEL_VALUE
    #                 elif player_research_points >= URAN_RESEARCH_POINTS - 50:
    #                     A1[5, x, y] = 1
    #             else:
    #                 raise ValueError
    #             A1[3, x, y] = fuel / FUEL_BOUND
    #         A1[4, x, y] = cell.road / MAX_ROAD

    # define city tiles, 0 or 1 for bool, 0 to around 1 for float;
    # layers:
    number_of_main_layers = 39
    A2 = np.zeros((number_of_main_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)

    # 0 - a unit
    # 1 - is player
    # 2 - is opponent
    # 3 - at the city tile
    # 4 - first place in the city tile is occupied by the unit - fill later (in get_separate_outputs)
    # 5 - second place is occupied by the unit, and the first was occupied before - fill later
    # 6 - third place is occupied - fill later
    # 7 - forth place is occupied - fill later
    # 8 - the place number is more than 4th - fill later

    # 9 - is worker - X0
    # 10 - is cart - X1
    # 11 - can act - X2
    # 12 - can build - X3
    # 13 - cargo wood - X4
    # 14 - cargo coal - X5
    # 15 - cargo uranium - X6
    # 16 - cargo space left - X7
    # 17 - fuel equivalent - X8

    # 18 - is city tile
    # 19 - is player
    # 20 - is opponent
    # 21 - can act
    # 22 - amount of city tiles in the city, which the city tile belongs to
    # 23 - current city upkeep
    # 24 - fuel amount
    # 25 - ratio if city can survive, 1 and more means it can

    # 26 - amount of all friendly city tiles
    # 27 - amount of cities
    # 28 - units build limit reached (workers + carts == city tiles)
    # 29 - number of workers
    # 30 - number of carts
    # 31 - number of friendly units
    # 32 - research progress for coal
    # 33 - research progress for uranium
    # 34 - progress (from 0 to 1) until next day
    # 35 - progress until next night
    # 36 - progress until finish
    # 37 - is night
    # 38 - current cycle

    # start with city tiles to know their positions to fill units cells
    for k, city in list(player.cities.items()) + list(opponent.cities.items()):
        if city.team == player.team:
            city_tiles_count = player_city_tiles_count
            cities_count = player_cities_count
            units_count = player_units_count
            workers_count = player_workers_count
            carts_count = player_carts_count
            research_points = player_research_points
        elif city.team == opponent.team:
            city_tiles_count = opponent_city_tiles_count
            cities_count = opponent_cities_count
            units_count = opponent_units_count
            workers_count = opponent_workers_count
            carts_count = opponent_carts_count
            research_points = opponent_research_points
        else:
            raise ValueError
        current_light_upkeep = city.get_light_upkeep()
        current_fuel = city.fuel
        current_city_tiles_count = 0
        for _ in city.citytiles:
            current_city_tiles_count += 1
        for city_tile in city.citytiles:
            # city tile group
            y, x = city_tile.pos.x + shift, city_tile.pos.y + shift
            A2[18, x, y] = 1
            if city_tile.team == player.team:
                A2[19, x, y] = 1
            elif city_tile.team == opponent.team:
                A2[20, x, y] = 1
            else:
                raise ValueError
            if city_tile.can_act():
                A2[21, x, y] = 1
                if city_tile.team == player.team:
                    player_city_tiles_coords[f"ct_{x}_{y}"] = (x, y)  # to save only the operable units
            A2[22, x, y] = current_city_tiles_count / CITY_TILES_IN_CITY_BOUND
            A2[23, x, y] = UPKEEP_BOUND_PER_TILE / current_light_upkeep
            A2[24, x, y] = current_fuel / FUEL_BOUND
            A2[25, x, y] = min(1, current_fuel / (min(10, to_next_day) * current_light_upkeep))  # ratio to survive

            # common group
            A2[26, x, y] = city_tiles_count / CITY_TILES_BOUND
            A2[27, x, y] = cities_count / CITIES_BOUND
            if units_count == city_tiles_count:
                A2[28, x, y] = 1
            A2[29, x, y] = workers_count / WORKERS_BOUND
            A2[30, x, y] = carts_count / CARTS_BOUND
            A2[31, x, y] = units_count / UNITS_BOUND
            A2[32, x, y] = min(research_points / COAL_RESEARCH_POINTS, 1)
            A2[33, x, y] = min(research_points / URAN_RESEARCH_POINTS, 1)
            A2[34, x, y] = 1 - to_next_day / CYCLE_LENGTH
            A2[35, x, y] = 1 - to_next_night / CYCLE_LENGTH
            A2[36, x, y] = turn / MAX_DAYS
            A2[37, x, y] = is_night
            A2[38, x, y] = current_cycle / TOTAL_CYCLES

    for unit in player.units + opponent.units:
        # unit group
        if unit.team == player.team:
            city_tiles_count = player_city_tiles_count
            cities_count = player_cities_count
            units_count = player_units_count
            workers_count = player_workers_count
            carts_count = player_carts_count
            research_points = player_research_points
        elif unit.team == opponent.team:
            city_tiles_count = opponent_city_tiles_count
            cities_count = opponent_cities_count
            units_count = opponent_units_count
            workers_count = opponent_workers_count
            carts_count = opponent_carts_count
            research_points = opponent_research_points
        else:
            raise ValueError
        y, x = unit.pos.x + shift, unit.pos.y + shift
        A2[0, x, y] = 1
        if unit.team == player.team:
            A2[1, x, y] = 1
        elif unit.team == opponent.team:
            A2[2, x, y] = 1
        else:
            raise ValueError
        is_unit_at_home = 1 if A2[18, x, y] == 1 else 0
        A2[3, x, y] = is_unit_at_home

        X = np.zeros(9, dtype=np.half)
        if unit.is_worker():
            X[0] = 1
            cargo_cap = WORKERS_CARGO
        elif unit.is_cart():
            X[1] = 1
            cargo_cap = CART_CARGO
        else:
            raise ValueError
        if unit.can_act():
            X[2] = 1
        if unit.can_build(current_game_state.map):
            X[3] = 1
        X[4] = unit.cargo.wood / cargo_cap
        X[5] = unit.cargo.coal / cargo_cap
        X[6] = unit.cargo.uranium / cargo_cap
        X[7] = unit.get_cargo_space_left() / cargo_cap
        X[8] = (unit.cargo.wood * WOOD_FUEL_VALUE +
                unit.cargo.coal * COAL_FUEL_VALUE +
                unit.cargo.uranium * URAN_FUEL_VALUE) / FUEL_BOUND

        # there are many unit can share the same position at home
        # so save unique unit parameters in X array and store it in dictionary if unit is at home
        # if unit is not at home so it has a unique position, put it inside A2 array
        if is_unit_at_home:
            if unit.can_act() and unit.team == player.team:
                player_units_coords[unit.id] = ((x, y), (X, unit.is_worker()))
        else:
            if unit.can_act() and unit.team == player.team:
                player_units_coords[unit.id] = ((x, y), (None, unit.is_worker()))

            A2[9:18, x, y] = X

            # common group
            A2[26, x, y] = city_tiles_count / CITY_TILES_BOUND
            A2[27, x, y] = cities_count / CITIES_BOUND
            if units_count == city_tiles_count:
                A2[28, x, y] = 1
            A2[29, x, y] = workers_count / WORKERS_BOUND
            A2[30, x, y] = carts_count / CARTS_BOUND
            A2[31, x, y] = units_count / UNITS_BOUND
            A2[32, x, y] = min(research_points / COAL_RESEARCH_POINTS, 1)
            A2[33, x, y] = min(research_points / URAN_RESEARCH_POINTS, 1)
            A2[34, x, y] = 1 - to_next_day / CYCLE_LENGTH
            A2[35, x, y] = 1 - to_next_night / CYCLE_LENGTH
            A2[36, x, y] = turn / MAX_DAYS
            A2[37, x, y] = is_night
            A2[38, x, y] = current_cycle / TOTAL_CYCLES

    A = np.concatenate((A2, A1), axis=0)

    # define headers
    # layers:
    # 0 - an operable one
    # 1 - is worker
    # 2 - is cart
    # 3 - is city tile
    # 4 - prev pos for units
    # 5 - prev prev pos for units
    number_of_header_layers = 6
    units_headers = {}
    if player_units_coords:
        for k, ((x, y), (X, is_worker)) in player_units_coords.items():
            head = np.zeros((number_of_header_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
            worker = np.array([1, 1, 0, 0], dtype=np.half)
            cart = np.array([1, 0, 1, 0], dtype=np.half)
            head[:4, x, y] = worker if is_worker else cart

            if k in units_actions_dict.keys():
                units_actions_dict[k].append((x, y))
                unit_prev_pos = units_actions_dict[k][-2]
                if len(units_actions_dict[k]) > 2:
                    unit_prev_prev_pos = units_actions_dict[k][-3]
                else:
                    unit_prev_prev_pos = units_actions_dict[k][-2]
            else:
                units_actions_dict[k] = []
                units_actions_dict[k].append((x, y))
                unit_prev_pos = (x, y)
                unit_prev_prev_pos = (x, y)
            head[4, unit_prev_pos[0], unit_prev_pos[1]] = 1
            head[5, unit_prev_prev_pos[0], unit_prev_prev_pos[1]] = 1

            head = np.moveaxis(head, 0, -1)
            units_headers[k] = (head, (x, y), X, is_worker)

    city_tiles_headers = {}
    if player_city_tiles_coords:
        for k, (x, y) in player_city_tiles_coords.items():
            head = np.zeros((number_of_header_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
            head[:4, x, y] = np.array([1, 0, 0, 1], dtype=np.half)
            head = np.moveaxis(head, 0, -1)
            city_tiles_headers[k] = head

    B = np.moveaxis(A, 0, -1)

    outputs = {"stem": B,
               "units_headers": units_headers,
               "city_tiles_headers": city_tiles_headers}

    return outputs


@tf.function
def squeeze(feature_layers_in):
    features = feature_layers_in
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
    # piece_glob4 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 45:49]
    # piece_glob5 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 50:]
    piece_glob4 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 45:54]
    piece_glob = tf.concat([piece_glob1, piece_glob2, piece_glob3, piece_glob4], axis=-1)
    # 17, 4; 5, 5
    pooled_piece_glob = tf.squeeze(tf.nn.avg_pool(tf.expand_dims(piece_glob, axis=0), 5, 5, padding="VALID"))
    # piece_glob_v = piece_glob.numpy()
    # pooled_piece_glob_v = pooled_piece_glob.numpy()

    piece_filtered = tf.concat([piece[:, :, :1],
                                piece[:, :, 4:10],
                                piece[:, :, 15:],
                                ], axis=-1)
    # piece_filtered_v = piece_filtered.numpy()
    observations = tf.concat([piece_filtered, pooled_piece_glob], axis=-1)
    return observations


def get_separate_outputs(observation, current_game_state):
    inputs = process(observation, current_game_state)
    stem = inputs["stem"]
    units_headers = inputs["units_headers"]
    city_tiles_headers = inputs["city_tiles_headers"]

    workers = {}
    carts = {}
    if units_headers:
        for k, header in units_headers.items():
            head, (x, y), X, is_worker = header
            mod_stem = np.copy(stem)
            order_stem = np.copy(stem[x, y, 4:9])
            if X is not None:
                mod_stem[x, y, 9:18] = X
                if order_stem[0] == 0:
                    order_stem[0] = 1
                elif order_stem[0] == 1:
                    if order_stem[1] == 0:
                        order_stem[1] = 1
                    elif order_stem[1] == 1:
                        if order_stem[2] == 0:
                            order_stem[2] = 1
                        elif order_stem[2] == 1:
                            if order_stem[3] == 0:
                                order_stem[3] = 1
                            elif order_stem[3] == 1:
                                order_stem[4] = 1
                            else:
                                raise ValueError
                        else:
                            raise ValueError
                    else:
                        raise ValueError
                else:
                    raise ValueError
                mod_stem[x, y, 4:9] = order_stem
            ready_array = np.concatenate((head, mod_stem), axis=-1)
            ready_array = squeeze(tf.constant(ready_array, dtype=tf.float32)).numpy().astype(dtype=np.half)
            if is_worker:
                workers[k] = ready_array
            else:
                carts[k] = ready_array

    city_tiles = {}
    if city_tiles_headers:
        for k, header in city_tiles_headers.items():
            ready_array = np.concatenate((header, stem), axis=-1)
            city_tiles[k] = ready_array

    outputs = {"workers": workers,
               "carts": carts,
               "city_tiles": city_tiles}
    return outputs


if __name__ == "__main__":
    test_get_timing()
