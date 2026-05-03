from typing import List, Optional

import numpy as np


def _time_axis(fs: int, duration: float) -> np.ndarray:
    return np.arange(int(fs * duration)) / fs



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


def generate_white_noise(
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    rng = np.random.default_rng(seed)
    return t, amplitude * rng.uniform(-1.0, 1.0, size=len(t))


def generate_pink_noise(
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # FFT-based 1/f shaping: scale each frequency bin by 1/sqrt(f)
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    fft = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    fft /= np.sqrt(freqs)
    fft[0] = 0.0
    pink = np.fft.irfft(fft, n=n)
    pink /= np.max(np.abs(pink)) + 1e-12
    return t, amplitude * pink


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


def generate_chirp(
    f_start: float = 20.0,
    f_end: float = 20000.0,
    fs: int = 44100,
    duration: float = 1.0,
    amplitude: float = 1.0,
    method: str = "logarithmic",
) -> tuple[np.ndarray, np.ndarray]:
    t = _time_axis(fs, duration)
    if method == "linear":
        phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    else:
        # Exponential sweep: constant ratio per octave, preferred for audio testing
        k = np.log(f_end / f_start) / duration
        phase = 2 * np.pi * f_start * (np.exp(k * t) - 1) / k
    return t, amplitude * np.sin(phase)


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
