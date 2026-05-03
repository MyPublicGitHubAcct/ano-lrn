import numpy as np


# ── Modulator ─────────────────────────────────────────────────────────────────

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
