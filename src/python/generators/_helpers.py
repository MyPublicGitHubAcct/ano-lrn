import numpy as np


def _time_axis(fs: int, duration: float) -> np.ndarray:
    return np.arange(int(fs * duration)) / fs
