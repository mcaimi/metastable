#!/usr/bin/env python
"""
Random Seed Helpers
"""

import random

# Random seed generation bit length
RANDOM_BIT_LENGTH = 32


def get_random_seed(seed_len: int = RANDOM_BIT_LENGTH) -> int:
    """
    Generate a random seed value.

    Args:
        seed_len: Number of random bits to generate

    Returns:
        Random non-negative integer seed
    """
    return random.getrandbits(seed_len)


def resolve_seed(seed: int) -> int:
    """
    Resolve a user-supplied seed: any negative value means "pick one at random".

    Args:
        seed: Requested seed (negative for random)

    Returns:
        A concrete, non-negative seed
    """
    if seed < 0:
        return get_random_seed(RANDOM_BIT_LENGTH)
    return seed
