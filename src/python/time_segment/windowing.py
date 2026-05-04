import numpy as np

_WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


def apply_window(signal: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """Multiply signal by a window function to reduce spectral leakage.

    window_type: 'hann' | 'hamming' | 'blackman' | 'rectangular'.
    """
    n = len(signal)
    if window_type == "rectangular":
        return signal.copy()
    if window_type not in _WINDOWS:
        raise ValueError(f"Unknown window type '{window_type}'. Choose from {list(_WINDOWS)} or 'rectangular'.")
    return signal * _WINDOWS[window_type](n)
