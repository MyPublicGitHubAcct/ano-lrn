import numpy as np

from python.delay.line import fractional_delay_line


def vibrato(
    signal: np.ndarray,
    rate: float,
    depth_samples: float,
    fs: int = 44100,
) -> np.ndarray:
    """Pitch modulation via sinusoidally varying fractional delay.

    depth_samples: peak delay modulation in samples (determines pitch deviation).
    """
    n = len(signal)
    t = np.arange(n) / fs
    center = int(depth_samples) + 1
    delay = center + depth_samples * np.sin(2 * np.pi * rate * t)
    return fractional_delay_line(signal, delay)
