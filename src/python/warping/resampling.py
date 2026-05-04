import numpy as np
from math import gcd

from scipy.signal import resample_poly


def resample(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Polyphase resampling from orig_fs to target_fs.

    Uses an integer up/down ratio derived from the GCD of the two rates.
    """
    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(signal, up, down).astype(float)
