import numpy as np
from scipy.signal import lfilter


def formant_filter(
    signal: np.ndarray,
    formant_freqs: list,
    bandwidths: list,
    fs: int = 44100,
) -> np.ndarray:
    """Cascade of 2nd-order resonators at each formant frequency.

    Each resonator boosts energy at formant_freqs[i] with bandwidth
    bandwidths[i] Hz (–3 dB full bandwidth).
    Pole radius: R = exp(-pi * bw / fs).
    """
    out = np.asarray(signal, dtype=float)
    for f0, bw in zip(formant_freqs, bandwidths):
        R = np.exp(-np.pi * max(bw, 1.0) / fs)
        theta = 2 * np.pi * f0 / fs
        b = [1.0 - R ** 2]
        a = [1.0, -2.0 * R * np.cos(theta), R ** 2]
        out = lfilter(b, a, out).astype(float)
    return out
