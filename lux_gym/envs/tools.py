import numpy as np

# constants from game rules
MAX_MAP_SIDE = 32
WOOD_FUEL_VALUE = 1
COAL_FUEL_VALUE = 5
URAN_FUEL_VALUE = 20
COAL_RESEARCH_POINTS = 50
URAN_RESEARCH_POINTS = 200
MAX_ROAD = 6
# constants for normalization
WOOD_BOUND = 500
COAL_BOUND = 500
URAN_BOUND = 500
FUEL_BOUND = 10000


def process(observation, current_game_state):
    """
    Args:
        observation: An observation, which agents get as an input from kaggle environment.
        current_game_state: An object provided by kaggle to simplify game info extraction.
    Returns:
        processed_observations: An observation, which a model can use as an input.
    """
    player = current_game_state.players[observation.player]
    player_research_points = player.research_points
    opponent = current_game_state.players[(observation.player + 1) % 2]
    width, height = current_game_state.map.width, current_game_state.map.height

    # define resources and roads, 0 or 1 for bool, 0 to 1 for float;
    # layers:
    number_of_resources_layers = 10
    A1 = np.zeros((number_of_resources_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
    # 0 - a resource
    # 1 - is wood
    # 2 - wood amount
    # 3 - is coal
    # 4 - coal amount
    # 5 - is uranium
    # 6 - uranium amount
    # 7 - fuel equivalent
    # 8 - if a resource is available, 1 when ready
    # 9 - a road lvl
    # maybe add the resource availability for an opponent?
    for y in range(height):
        for x in range(width):
            cell = current_game_state.map.get_cell(x, y)
            if cell.has_resource():
                A1[0, x, y] = 1  # a resource at the point
                resource = cell.resource
                if resource.type == "wood":
                    A1[1, x, y] = 1
                    wood_amount = resource.amount
                    A1[2, x, y] = min(wood_amount / WOOD_BOUND, 1)
                    fuel = wood_amount * WOOD_FUEL_VALUE
                    A1[8, x, y] = 1  # wood is always available
                elif resource.type == "coal":
                    A1[3, x, y] = 1
                    coal_amount = resource.amount
                    A1[4, x, y] = min(coal_amount / COAL_BOUND, 1)
                    fuel = coal_amount * COAL_FUEL_VALUE
                    A1[8, x, y] = min(player_research_points / COAL_RESEARCH_POINTS, 1)
                elif resource.type == "uranium":
                    A1[5, x, y] = 1
                    uran_amount = resource.amount
                    A1[6, x, y] = min(uran_amount / URAN_BOUND, 1)
                    fuel = uran_amount * URAN_FUEL_VALUE
                    A1[8, x, y] = min(player_research_points / URAN_RESEARCH_POINTS, 1)
                else:
                    raise ValueError
                A1[7, x, y] = min(fuel / FUEL_BOUND, 1)
                A1[9, x, y] = cell.road / MAX_ROAD

    # define units, 0 or 1 for bool, 0 to 1 for float;
    # layers:
    number_of_units_layers = 9
    A2 = np.zeros((number_of_units_layers, MAX_MAP_SIDE, MAX_MAP_SIDE), dtype=np.half)
    # 0 - a unit
    # 1 - is player
    # 2 - is opponent
    # 3 - can act
    # 4 - is city tile
    # 5 - amount of city tiles
    # 6 - is worker
    # 7 - number of workers
    # 8 - is cart
    # 9 - number of carts
    # 10 - research progress for coal
    # 11 - research progress for uranium

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

    reversed_order = ((d[:, None] & (1 << np.arange(m))) > 0).astype(np.uint8)
    return np.fliplr(reversed_order)
