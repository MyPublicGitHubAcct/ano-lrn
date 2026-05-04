import numpy as np


def vibrato(
    signal: np.ndarray,
    rate: float,
    depth_samples: float,
    fs: int = 44100,
) -> np.ndarray:
    """Pitch modulation via sinusoidally varying fractional delay.

    depth_samples: peak delay modulation in samples (determines pitch deviation).
    Uses linear interpolation for sub-sample accuracy.
    """
    n = len(signal)
    t = np.arange(n) / fs
    center = int(depth_samples) + 1
    delay = center + depth_samples * np.sin(2 * np.pi * rate * t)
    pad = center + int(depth_samples) + 2
    padded = np.concatenate([np.zeros(pad), signal])
    out = np.zeros(n)
    for i in range(n):
        d = delay[i]
        d_int = int(d)
        frac = d - d_int
        idx = pad + i
        out[i] = padded[idx - d_int] * (1.0 - frac) + padded[idx - d_int - 1] * frac
    return out
