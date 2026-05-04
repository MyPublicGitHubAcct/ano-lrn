import numpy as np

from python.delay import delay_line


def haas(
    signal: np.ndarray,
    delay_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Haas (precedence) effect: left channel is dry, right is slightly delayed.

    Small delays (< ~40 ms) create spatial width without apparent echo.
    Returns (left, right) arrays.
    """
    return signal.copy(), delay_line(signal, delay_samples)
