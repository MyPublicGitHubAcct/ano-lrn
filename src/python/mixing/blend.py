import numpy as np


def mix(signals: list, weights: list = None) -> np.ndarray:
    """Weighted sum of signals.

    If weights is None, all signals are summed with equal weight 1/N.
    All signals must have the same length.
    """
    arrays = [np.asarray(s, dtype=float) for s in signals]
    if weights is None:
        w = [1.0 / len(arrays)] * len(arrays)
    else:
        w = list(weights)
    out = np.zeros_like(arrays[0])
    for s, wt in zip(arrays, w):
        out = out + wt * s
    return out


def crossfade(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    position: float,
) -> np.ndarray:
    """Linear crossfade between two signals.

    position = 0.0: fully signal_a; position = 1.0: fully signal_b.
    """
    p = float(np.clip(position, 0.0, 1.0))
    return np.asarray(signal_a, dtype=float) * (1.0 - p) + np.asarray(signal_b, dtype=float) * p
