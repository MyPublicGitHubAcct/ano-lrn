import numpy as np


def diode_clip(signal: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Asymmetric diode-rectifier clipping.

    Positive half: hard clipped at +threshold.
    Negative half: soft clipped via tanh (models a forward-biased diode
    that turns on gradually).
    """
    if threshold == 0.0:
        return np.zeros(len(signal), dtype=float)
    out = signal.copy().astype(float)
    out = np.where(out > threshold, threshold, out)
    neg = out < 0
    out[neg] = -threshold * np.tanh(-out[neg] / threshold)
    return out


def wavefold(signal: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Wavefolder: reflects signal at ±1 boundaries instead of clipping.

    Multiplies by gain then mirrors back each time the value would exceed ±1.
    Higher gain crosses the boundary more times, adding harmonics on each fold.
    """
    x = np.asarray(signal, dtype=float) * gain
    # Triangle-wave reflection: fold x into [-1, 1] using modular arithmetic.
    # Period-4 triangle: map (x+1) mod 4 → rising half [0,2) gives x-1, falling [2,4) gives 3-x.
    x_mod = np.mod(x + 1.0, 4.0)
    return np.where(x_mod < 2.0, x_mod - 1.0, 3.0 - x_mod)


def analog_saturate(signal: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """3rd-order polynomial soft saturation (odd harmonics, tube-style).

    Transfer function: y = x - x^3 / 3, applied after `drive` gain.
    Input is clamped to [-1, 1] before the polynomial to bound the output.
    """
    x = np.clip(np.asarray(signal, dtype=float) * drive, -1.0, 1.0)
    return x - x ** 3 / 3.0
