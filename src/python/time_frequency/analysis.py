import numpy as np

from python.time_frequency.transform import stft


def spectrogram(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop_size: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Magnitude spectrogram in dB.

    Returns a 2D array of shape (frame_size // 2 + 1, num_frames).
    """
    S = stft(signal, frame_size, hop_size, window)
    return 20.0 * np.log10(np.abs(S) + 1e-12)
