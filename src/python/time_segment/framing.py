import numpy as np

_WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


def frame(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Segment signal into overlapping frames.

    Returns a 2D array of shape (num_frames, frame_size).
    Frames beyond the signal end are zero-padded.
    """
    n = len(signal)
    num_frames = max(1, 1 + (n - frame_size) // hop_size)
    out = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        chunk = signal[start : start + frame_size]
        out[i, : len(chunk)] = chunk
    return out


def overlap_add(
    frames: np.ndarray,
    hop_size: int,
    window_type: str = "hann",
) -> np.ndarray:
    """Reconstruct a signal from overlapping frames using the OLA method.

    frames: 2D array of shape (num_frames, frame_size).
    The normalisation window compensates for the analysis window applied to
    the frames before synthesis.
    """
    num_frames, frame_size = frames.shape
    out_len = (num_frames - 1) * hop_size + frame_size
    out = np.zeros(out_len)
    norm = np.zeros(out_len)

    if window_type == "rectangular":
        win = np.ones(frame_size)
    elif window_type in _WINDOWS:
        win = _WINDOWS[window_type](frame_size)
    else:
        raise ValueError(f"Unknown window type '{window_type}'.")

    for i, frm in enumerate(frames):
        start = i * hop_size
        out[start : start + frame_size] += frm * win
        norm[start : start + frame_size] += win ** 2

    norm = np.where(norm > 1e-8, norm, 1.0)
    return out / norm
