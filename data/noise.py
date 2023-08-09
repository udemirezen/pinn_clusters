import numpy as np

def add_noise(data: np.array, ratio: float):
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Noise ratio must within [0, 1], got {ratio}")
    noise = np.random.normal(0, ratio, data.shape)
    return data + data * noise
