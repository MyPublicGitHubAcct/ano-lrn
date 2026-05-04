import numpy as np
from scipy.ndimage import median_filter

from python.time_frequency import istft, stft


def _match_length(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) >= length:
        return x[:length]
    return np.concatenate([x, np.zeros(length - len(x))])


def hpss(
    signal: np.ndarray,
    fs: int = 44100,
    frame_size: int = 2048,
    hop_size: int = 512,
    kernel_size: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-percussive source separation via median filtering (Fitzgerald 2010).

    Harmonic component: median-filtered along the time axis (horizontal).
    Percussive component: median-filtered along the frequency axis (vertical).
    Separation uses Wiener-style soft masks: H^2 / (H^2 + P^2).
    Returns (harmonic, percussive), both the same length as `signal`.
    """
    S = stft(signal, frame_size, hop_size)
    mag = np.abs(S)
    phase = np.angle(S)

    H_mag = median_filter(mag, size=(1, kernel_size))
    P_mag = median_filter(mag, size=(kernel_size, 1))

    denom = H_mag ** 2 + P_mag ** 2 + 1e-12
    H_mask = H_mag ** 2 / denom
    P_mask = P_mag ** 2 / denom

    harmonic = istft(H_mask * mag * np.exp(1j * phase), hop_size)
    percussive = istft(P_mask * mag * np.exp(1j * phase), hop_size)

    n = len(signal)
    return _match_length(harmonic, n), _match_length(percussive, n)
