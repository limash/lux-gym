import math

import numpy as np

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
WORKERS_BOUND = 50
CARTS_BOUND = 50
UNITS_BOUND = WORKERS_BOUND + CARTS_BOUND
CITY_TILES_BOUND = UNITS_BOUND  # since the units number is limited by the city tiles number
CITY_TILES_IN_CITY_BOUND = 25
# from https://www.kaggle.com/c/lux-ai-2021/discussion/265886
UPKEEP_BOUND = 10 * CITY_TILES_IN_CITY_BOUND + 20 * math.sqrt(CITY_TILES_IN_CITY_BOUND)
CITIES_BOUND = 5


def process(observation, current_game_state):
    """
    Args:
        observation: An observation, which agents get as an input from kaggle environment.
        current_game_state: An object provided by kaggle to simplify game info extraction.
    Returns:
        processed_observations: An observation, which a model can use as an input.
    """
    player = current_game_state.players[observation.player]
    opponent = current_game_state.players[(observation.player + 1) % 2]
    width, height = current_game_state.map.width, current_game_state.map.height
    turn = current_game_state.turn

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
    number_of_resources_layers = 20
    A1 = np.zeros((number_of_resources_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
    for y in range(height):
        for x in range(width):
            cell = current_game_state.map.get_cell(x, y)
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
            A1[10:15, x, y] = to_binary(np.asarray((x,), dtype=np.uint8), m=5)
            A1[15:20, x, y] = to_binary(np.asarray((y,), dtype=np.uint8), m=5)

    # define units, 0 or 1 for bool, 0 to 1 for float;
    # layers:
    number_of_units_layers = 40
    A2 = np.zeros((number_of_units_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
    # 0 - a unit

    # 1 - is worker
    # 2 - is player
    # 3 - is opponent
    # 4 - can act
    # 5 - can build
    # 6 - cargo wood
    # 7 - cargo coal
    # 8 - cargo uranium
    # 9 - cargo space left
    # 10 - fuel equivalent

    # 11 - is cart
    # 12 - is player
    # 13 - is opponent
    # 14 - can act
    # 15 - cargo wood
    # 16 - cargo coal
    # 17 - cargo uranium
    # 18 - cargo space left
    # 19 - fuel equivalent

    # 20 - is city tile
    # 21 - is player
    # 22 - is opponent
    # 23 - can act
    # 24 - amount of city tiles in the city, which the city tile belongs to
    # 25 - current city upkeep
    # 26 - fuel amount

    # 27 - amount of all friendly city tiles
    # 28 - amount of cities
    # 29 - units build limit reached (workers + carts == city tiles)
    # 30 - number of workers
    # 31 - number of carts
    # 32 - number of friendly units
    # 33 - research progress for coal
    # 34 - research progress for uranium
    # 35 - progress (from 0 to 1) until next day
    # 36 - progress until next night
    # 37 - progress until finish
    # 38 - is night
    # 39 - current cycle

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
        x, y = unit.pos.x, unit.pos.y
        A2[0, x, y] = 1
        if unit.is_worker():
            # worker group
            A2[1, x, y] = 1
            if unit.team == player.team:
                A2[2, x, y] = 1
            elif unit.team == opponent.team:
                A2[3, x, y] = 1
            else:
                raise ValueError
            if unit.can_act():
                A2[4, x, y] = 1
            if unit.can_build(current_game_state.map):
                A2[5, x, y] = 1
            A2[6, x, y] = unit.cargo.wood / WORKERS_CARGO
            A2[7, x, y] = unit.cargo.coal / WORKERS_CARGO
            A2[8, x, y] = unit.cargo.uranium / WORKERS_CARGO
            A2[9, x, y] = unit.get_cargo_space_left() / WORKERS_CARGO
            A2[10, x, y] = (unit.cargo.wood * WOOD_FUEL_VALUE +
                            unit.cargo.coal * COAL_FUEL_VALUE +
                            unit.cargo.uranium * URAN_FUEL_VALUE) / FUEL_BOUND
        elif unit.is_cart():
            # cart group
            A2[11, x, y] = 1
            if unit.team == player.team:
                A2[12, x, y] = 1
            elif unit.team == opponent.team:
                A2[13, x, y] = 1
            else:
                raise ValueError
            if unit.can_act():
                A2[14, x, y] = 1
            A2[15, x, y] = unit.cargo.wood / CART_CARGO
            A2[16, x, y] = unit.cargo.coal / CART_CARGO
            A2[17, x, y] = unit.cargo.uranium / CART_CARGO
            A2[18, x, y] = unit.get_cargo_space_left() / CART_CARGO
            A2[19, x, y] = (unit.cargo.wood * WOOD_FUEL_VALUE +
                            unit.cargo.coal * COAL_FUEL_VALUE +
                            unit.cargo.uranium * URAN_FUEL_VALUE) / FUEL_BOUND

        # common group
        A2[27, x, y] = city_tiles_count / CITY_TILES_BOUND
        A2[28, x, y] = cities_count / CITIES_BOUND
        if units_count == city_tiles_count:
            A2[29, x, y] = 1
        A2[30, x, y] = workers_count / WORKERS_BOUND
        A2[31, x, y] = carts_count / CARTS_BOUND
        A2[32, x, y] = units_count / UNITS_BOUND

        A2[33, x, y] = min(research_points / COAL_RESEARCH_POINTS, 1)
        A2[34, x, y] = min(research_points / URAN_RESEARCH_POINTS, 1)

        A2[35, x, y] = to_next_day / CYCLE_LENGTH
        A2[36, x, y] = to_next_night / CYCLE_LENGTH
        A2[37, x, y] = turn / MAX_DAYS
        A2[38, x, y] = is_night
        A2[39, x, y] = current_cycle / TOTAL_CYCLES

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
            x, y = city_tile.pos.x, city_tile.pos.y
            A2[20, x, y] = 1
            if city_tile.team == player.team:
                A2[21, x, y] = 1
            elif city_tile.team == opponent.team:
                A2[22, x, y] = 1
            else:
                raise ValueError
            if city_tile.can_act():
                A2[23, x, y] = 1
            A2[24, x, y] = current_city_tiles_count / CITY_TILES_IN_CITY_BOUND
            A2[25, x, y] = current_light_upkeep / UPKEEP_BOUND
            A2[26, x, y] = current_fuel / FUEL_BOUND

            # common group
            A2[27, x, y] = city_tiles_count / CITY_TILES_BOUND
            A2[28, x, y] = cities_count / CITIES_BOUND
            if units_count == city_tiles_count:
                A2[29, x, y] = 1
            A2[30, x, y] = workers_count / WORKERS_BOUND
            A2[31, x, y] = carts_count / CARTS_BOUND
            A2[32, x, y] = units_count

            A2[33, x, y] = min(research_points / COAL_RESEARCH_POINTS, 1)
            A2[34, x, y] = min(research_points / URAN_RESEARCH_POINTS, 1)

            A2[35, x, y] = to_next_day / CYCLE_LENGTH
            A2[36, x, y] = to_next_night / CYCLE_LENGTH
            A2[37, x, y] = turn / MAX_DAYS
            A2[38, x, y] = is_night
            A2[39, x, y] = current_cycle / TOTAL_CYCLES

    processed_observation = None
    return processed_observation


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
