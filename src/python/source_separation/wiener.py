import numpy as np

from python.time_frequency import istft, stft


def _match_length(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) >= length:
        return x[:length]
    return np.concatenate([x, np.zeros(length - len(x))])


def wiener_filter(
    mixture: np.ndarray,
    source_estimate: np.ndarray,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """Wiener filter: extract source from mixture given a spectral estimate.

    Applies a time-frequency mask = |estimate|^2 / |mixture|^2 (clamped to 1).
    Commonly used for speech enhancement and denoising.
    Returns a signal the same length as `mixture`.
    """
    M = stft(mixture, frame_size, hop_size)
    E = stft(source_estimate, frame_size, hop_size)
    mask = np.minimum(np.abs(E) ** 2 / (np.abs(M) ** 2 + 1e-12), 1.0)
    out = istft(mask * M, hop_size)
    return _match_length(out, len(mixture))
