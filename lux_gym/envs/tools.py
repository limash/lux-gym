import numpy as np


def process(observation, current_game_state):
    """
    Args:
        observation: An observation, which agents get as an input from kaggle environment.
        current_game_state: An object provided by kaggle to simplify game info extraction.
    Returns:
        processed_observations: An observation, which a model can use as an input.
    """
    max_map_side = 32
    player = current_game_state.players[observation.player]
    player_research_points = player.research_points
    opponent = current_game_state.players[(observation.player + 1) % 2]
    width, height = current_game_state.map.width, current_game_state.map.height

    # define resources, 0 or 1 for bool, 0 to 1 for float;
    # layers:
    number_of_resources_layers = 9
    # 0 - a resource
    # 1 - is wood
    # 2 - wood amount
    # 3 - is coal
    # 4 - coal amount
    # 5 - is uranium
    # 6 - uranium amount
    # 7 - fuel equivalent
    # 8 - if a resource is available
    # maybe add the resource availability for an opponent?
    wood_fuel_value = 1
    coal_fuel_value = 5
    uran_fuel_value = 20
    wood_amount, wood_bound = 0, 500  # 1 all for any amount more than 500
    coal_amount, coal_bound = 0, 500
    uran_amount, uran_bound = 0, 500
    fuel_bound = 10000
    A1 = np.zeros((number_of_resources_layers, max_map_side, max_map_side), dtype=np.half)
    for y in range(height):
        for x in range(width):
            cell = current_game_state.map.get_cell(x, y)
            if cell.has_resource():
                A1[0, x, y] = 1  # a resource at the point
                resource = cell.resource
                if resource.type == "wood":
                    A1[1, x, y] = 1
                    wood_amount = resource.amount
                    A1[2, x, y] = min(wood_amount / wood_bound, 1)
                    fuel = wood_amount * wood_fuel_value
                    A1[8, x, y] = 1  # wood is always available
                elif resource.type == "coal":
                    A1[3, x, y] = 1
                    coal_amount = resource.amount
                    A1[4, x, y] = min(coal_amount / coal_bound, 1)
                    fuel = coal_amount * coal_fuel_value
                    if player_research_points >= 50:
                        A1[8, x, y] = 1  # coal is available
                elif resource.type == "uranium":
                    A1[5, x, y] = 1
                    uran_amount = resource.amount
                    A1[6, x, y] = min(uran_amount / uran_bound, 1)
                    fuel = uran_amount * uran_fuel_value
                    if player_research_points >= 200:
                        A1[8, x, y] = 1  # uranium is available
                else:
                    raise ValueError
                A1[7, x, y] = min(fuel / fuel_bound, 1)

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
