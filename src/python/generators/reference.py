import numpy as np


def generate_dc(
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    return np.arange(n) / fs, np.full(n, amplitude)


def generate_nyquist(
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n) / fs
    return t, amplitude * -np.cos(np.pi * np.arange(n))


def generate_half_nyquist(
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n) / fs
    return t, amplitude * -np.cos(np.pi / 2 * np.arange(n))


def generate_quarter_nyquist(
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n) / fs
    return t, amplitude * -np.cos(np.pi / 4 * np.arange(n))
