import numpy as np


def tremolo(
    signal: np.ndarray,
    rate: float,
    depth: float,
    fs: int = 44100,
) -> np.ndarray:
    """Amplitude modulation via sinusoidal LFO.

    depth = 0: no modulation; depth = 1: amplitude oscillates between 0 and 1.
    """
    t = np.arange(len(signal)) / fs
    lfo = 1.0 - depth * 0.5 * (1.0 - np.cos(2 * np.pi * rate * t))
    return signal * lfo


def ring_modulate(
    signal: np.ndarray,
    carrier_freq: float,
    fs: int = 44100,
) -> np.ndarray:
    """Multiply signal by a cosine at carrier_freq Hz.

    Produces sum and difference sidebands; suppresses the carrier.
    """
    t = np.arange(len(signal)) / fs
    return signal * np.cos(2 * np.pi * carrier_freq * t)
