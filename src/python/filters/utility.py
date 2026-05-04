import numpy as np
from scipy.signal import lfilter


def dc_block(
    signal: np.ndarray,
    cutoff: float = 20.0,
    fs: int = 44100,
) -> np.ndarray:
    """First-order DC blocking filter.

    Removes DC offset and sub-sonic content below `cutoff` Hz.
    Uses the bilinear transform of a 1st-order analog high-pass, so the
    −3 dB point lands exactly at `cutoff` Hz.

    Transfer function: H(z) = (1 − z⁻¹) / (1 + k − (1 − k)·z⁻¹)
    where k = tan(π·cutoff/fs).  H(1) = 0 exactly (structural zero at DC).
    """
    k = np.tan(np.pi * cutoff / fs)
    b = [1.0 / (1.0 + k), -1.0 / (1.0 + k)]
    a = [1.0, -(1.0 - k) / (1.0 + k)]
    return lfilter(b, a, signal).astype(float)
