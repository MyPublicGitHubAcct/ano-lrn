import numpy as np


def moog_ladder(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    resonance: float = 0.0,
) -> np.ndarray:
    """Discretised Moog ladder filter (Huovilainen approximation).

    Four cascaded one-pole stages with nonlinear tanh saturation and a
    resonance feedback path.  resonance in [0, 1); values ≥ 1 cause
    self-oscillation.
    """
    fc = min(max(cutoff / fs, 1e-6), 0.499)
    g = 1.0 - np.exp(-2 * np.pi * fc)
    k = 4.0 * resonance
    out = np.zeros(len(signal))
    s = np.zeros(4)
    for i, x in enumerate(signal):
        x_fb = x - k * s[3]
        s[0] += g * (np.tanh(x_fb) - np.tanh(s[0]))
        s[1] += g * (np.tanh(s[0]) - np.tanh(s[1]))
        s[2] += g * (np.tanh(s[1]) - np.tanh(s[2]))
        s[3] += g * (np.tanh(s[2]) - np.tanh(s[3]))
        out[i] = s[3]
    return out
