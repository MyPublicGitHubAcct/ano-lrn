import numpy as np


def generate_adsr(
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.2,
    fs: int = 44100,
    duration: float = 1.0,
    curve: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n) / fs

    n_a = min(int(attack * fs), n)
    n_d = min(int(decay * fs), n - n_a)
    n_r = min(int(release * fs), n - n_a - n_d)
    n_s = n - n_a - n_d - n_r

    def _ramp(length: int, start: float, end: float, shaped: bool = False) -> np.ndarray:
        if length == 0:
            return np.empty(0)
        if not shaped or start == end:
            return np.linspace(start, end, length)
        # Normalized exponential that hits start and end exactly
        k = 5.0
        u = np.linspace(0.0, 1.0, length)
        c = (1.0 - np.exp(-k * u)) / (1.0 - np.exp(-k))
        return start + (end - start) * c

    use_exp = curve == "exponential"
    segments = [
        _ramp(n_a, 0.0, 1.0),
        _ramp(n_d, 1.0, sustain, shaped=use_exp),
        np.full(n_s, sustain),
        _ramp(n_r, sustain, 0.0, shaped=use_exp),
    ]
    env = np.concatenate([s for s in segments if len(s) > 0])
    return t, env


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
