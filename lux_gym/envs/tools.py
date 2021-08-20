import numpy as np


def process(observation, game_state):
    """
    Args:
        observation: An observation, which agents get as an input from kaggle environment.
        game_state: An object provided by kaggle to simplify game info extraction.
    Returns:
        processed_observations: An observation, which a model can use as an input.
    """

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
