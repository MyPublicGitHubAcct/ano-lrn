import numpy as np
from scipy.signal import lfilter


def feedback_delay(
    signal: np.ndarray,
    delay_samples: int,
    feedback: float = 0.5,
) -> np.ndarray:
    """IIR feedback comb filter: y[n] = x[n] + feedback * y[n - D].

    feedback > 0 produces echoes that decay exponentially.
    |feedback| must be < 1 for stability.
    """
    d = max(1, int(delay_samples))
    a = np.zeros(d + 1)
    a[0] = 1.0
    a[d] = -feedback
    return lfilter([1.0], a, signal).astype(float)


def comb_filter(
    signal: np.ndarray,
    delay_samples: int,
    gain: float = 0.5,
) -> np.ndarray:
    """FIR feedforward comb filter: y[n] = x[n] + gain * x[n - D].

    Creates peaks and notches spaced by fs / D Hz.
    """
    d = max(1, int(delay_samples))
    b = np.zeros(d + 1)
    b[0] = 1.0
    b[d] = gain
    return lfilter(b, [1.0], signal).astype(float)
