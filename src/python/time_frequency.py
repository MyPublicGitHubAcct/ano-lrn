import numpy as np


# ── Time-frequency ────────────────────────────────────────────────────────────

def stft(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop_size: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Short-time Fourier transform.

    Returns a complex array of shape (frame_size // 2 + 1, num_frames).
    Columns are time frames; rows are frequency bins from DC to Nyquist.
    """
    n = len(signal)
    win = np.hanning(frame_size) if window == "hann" else np.ones(frame_size)
    num_frames = max(1, 1 + (n - frame_size) // hop_size)
    S = np.zeros((frame_size // 2 + 1, num_frames), dtype=complex)
    for i in range(num_frames):
        start = i * hop_size
        frm = np.zeros(frame_size)
        chunk = signal[start : start + frame_size]
        frm[: len(chunk)] = chunk
        S[:, i] = np.fft.rfft(frm * win)
    return S


def istft(
    S: np.ndarray,
    hop_size: int = 512,
    window: str = "hann",
) -> np.ndarray:
    """Inverse STFT via overlap-add.

    S: complex array of shape (freq_bins, num_frames) as returned by stft().
    """
    freq_bins, num_frames = S.shape
    frame_size = (freq_bins - 1) * 2
    win = np.hanning(frame_size) if window == "hann" else np.ones(frame_size)
    out_len = (num_frames - 1) * hop_size + frame_size
    out = np.zeros(out_len)
    norm = np.zeros(out_len)
    for i in range(num_frames):
        start = i * hop_size
        frm = np.fft.irfft(S[:, i], n=frame_size)
        out[start : start + frame_size] += frm * win
        norm[start : start + frame_size] += win ** 2
    norm = np.where(norm > 1e-8, norm, 1.0)
    return out / norm


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
