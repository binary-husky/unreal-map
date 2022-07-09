"""Utilities for ensuring that experiments are deterministic."""
import random
import numpy as np
import tensorflow as tf

seed_ = None


def set_seed(seed):
    """Set the process-wide random seed.
    Args:
        seed (int): A positive integer
    """
    seed %= 234234234523
    global seed_
    seed_ = seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_seed():
    """Get the process-wide random seed.
    Returns:
        int: The process-wide random seed
    """
    return seed_
