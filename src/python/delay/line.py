import numpy as np


def delay_line(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Integer sample delay (pure FIR all-pass at z^-D)."""
    d = max(0, int(delay_samples))
    out = np.zeros_like(signal)
    if d < len(signal):
        out[d:] = signal[: len(signal) - d]
    return out
