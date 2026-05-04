import numpy as np


def generate_impulse(
    fs: int = 44100,
    duration: float = 1.0,
    delay: float = 0.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n) / fs
    wave = np.zeros(n)
    idx = int(delay * fs)
    if 0 <= idx < n:
        wave[idx] = amplitude
    return t, wave


def generate_step(
    fs: int = 44100,
    duration: float = 1.0,
    onset: float = 0.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n) / fs
    wave = np.zeros(n)
    idx = int(onset * fs)
    if idx < n:
        wave[idx:] = amplitude
    return t, wave
