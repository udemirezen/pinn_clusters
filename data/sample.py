"""
Functions for sampling data
"""
import numpy as np

def random_sample(n: int, x_star, *args):
    """
    Args:
        n: number of data to sample
        x_star: the x-location of training data to be sampled
        *args: rest of the training data (i.e. u, v, h) to be sampled
    """
    assert x_star.shape[0] == 401
    for arg in args:
        assert arg.shape[-1] == 401

    result = []
    if n < 1:
        raise ValueError(f"Invalid value for sampling data: try sample {n} from 401 data poitns")
    elif n == 1:
        for arg in args:
            result.append(arg[:, [0]])
        return x_star[[0]], *result
    else:
        sample_n = n - 1
        sample_range = np.arange(1, 401)
        np.random.shuffle(sample_range)
        sampled = [0] + list(sample_range[:sample_n])
        sampled.sort()

    for arg in args:
        result.append(arg[..., sampled])
    return x_star[sampled], *result
