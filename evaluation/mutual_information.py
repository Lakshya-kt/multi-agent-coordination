import numpy as np


def calculate_mutual_information(messages, directions):
    """
    TODO:
    Calculate I(M;D), where:

    M = discrete message (0, 1, 2, 3)
    D = direction of nearest visible target
        (up, down, left, right, none)

    Use empirical joint and marginal probabilities.
    Use log base 2 so the result is in bits.
    """
    raise NotImplementedError


def bootstrap_confidence_interval(values, num_bootstraps=1000):
    """
    TODO:
    Calculate a 95% bootstrap confidence interval.

    Resample at the appropriate independent unit,
    such as episodes or training seeds.
    """
    raise NotImplementedError