import numpy as np


def gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in dB (positive = amplify, negative = attenuate)."""
    return signal * 10.0 ** (gain_db / 20.0)


def normalize(signal: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Scale signal so its peak amplitude matches target_db.

    Silent signals (peak < 1e-12) are returned unchanged to avoid division
    by near-zero.
    """
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return signal.copy()
    target = 10.0 ** (target_db / 20.0)
    return signal * target / peak
