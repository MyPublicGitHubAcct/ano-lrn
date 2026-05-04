import numpy as np
from scipy.signal import lfilter


def _biquad(signal: np.ndarray, b: list, a: list) -> np.ndarray:
    return lfilter(b, a, signal).astype(float)
