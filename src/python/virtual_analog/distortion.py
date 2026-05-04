import numpy as np


def diode_clip(signal: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Asymmetric diode-rectifier clipping.

    Positive half: hard clipped at +threshold.
    Negative half: soft clipped via tanh (models a forward-biased diode
    that turns on gradually).
    """
    out = signal.copy().astype(float)
    out = np.where(out > threshold, threshold, out)
    neg = out < 0
    out[neg] = -threshold * np.tanh(-out[neg] / threshold)
    return out


def analog_saturate(signal: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """3rd-order polynomial soft saturation (odd harmonics, tube-style).

    Transfer function: y = x - x^3 / 3, applied after `drive` gain.
    Input is clamped to [-1, 1] before the polynomial to bound the output.
    """
    x = np.clip(np.asarray(signal, dtype=float) * drive, -1.0, 1.0)
    return x - x ** 3 / 3.0
