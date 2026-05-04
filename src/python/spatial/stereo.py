import numpy as np


def pan(
    signal: np.ndarray,
    position: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power stereo panning.

    position: -1.0 = hard left, 0.0 = centre, +1.0 = hard right.
    Returns (left, right) arrays. Left^2 + Right^2 = Signal^2 at all positions.
    """
    theta = (position + 1.0) * np.pi / 4.0  # maps [-1, 1] → [0, π/2]
    left = signal * np.cos(theta)
    right = signal * np.sin(theta)
    return left, right


def stereo_widen(
    left: np.ndarray,
    right: np.ndarray,
    width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """M/S stereo widening.

    width = 0: mono (side = 0); width = 1: unchanged; width > 1: wider.
    Returns (new_left, new_right).
    """
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * width
    return mid + side, mid - side
