from typing import Optional

import numpy as np

from python.generators._helpers import _time_axis


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
