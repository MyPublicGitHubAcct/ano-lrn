import numpy as np


def hard_clip(signal: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Clip signal symmetrically to [-threshold, threshold]."""
    return np.clip(signal, -threshold, threshold)


def soft_clip(signal: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh saturation; output stays in (-1, 1) regardless of drive."""
    return np.tanh(signal * drive)
