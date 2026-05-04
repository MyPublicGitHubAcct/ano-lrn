from typing import List

import numpy as np

from python.generators._helpers import _time_axis


def generate_sine(
    freq: float = 440.0,
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    return t, amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_square(
    freq: float = 440.0,
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
    duty: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    phase = (2 * np.pi * freq * t) % (2 * np.pi)
    wave = amplitude * np.where(phase < duty * 2 * np.pi, 1.0, -1.0)
    return t, wave


def generate_sawtooth(
    freq: float = 440.0,
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    wave = amplitude * (2 * ((freq * t) % 1.0) - 1)
    return t, wave


def generate_triangle(
    freq: float = 440.0,
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    phase = (freq * t) % 1.0
    wave = amplitude * (2 * np.abs(2 * phase - 1) - 1)
    return t, wave


def generate_multi_tone(
    freqs: List[float],
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    wave = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
    peak = np.max(np.abs(wave)) + 1e-12
    return t, amplitude * wave / peak
