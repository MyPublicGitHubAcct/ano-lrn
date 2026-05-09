import numpy as np


def delay_line(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Integer sample delay (pure FIR all-pass at z^-D)."""
    d = max(0, int(delay_samples))
    out = np.zeros_like(signal)
    if d < len(signal):
        out[d:] = signal[: len(signal) - d]
    return out


def fractional_delay_line(
    signal: np.ndarray,
    delay_samples,
    order: int = 3,
) -> np.ndarray:
    """Fractional sample delay via Lagrange polynomial interpolation.

    delay_samples: non-negative float scalar or 1-D array of floats (one per
        output sample). Fractional part drives sub-sample interpolation.
    order: Lagrange polynomial degree — 3 (4-point stencil) or 4 (5-point).
    """
    if order not in (3, 4):
        raise ValueError(f"order must be 3 or 4, got {order}")

    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    delay = np.asarray(delay_samples, dtype=float)
    if delay.ndim == 0:
        delay = np.full(N, float(delay))

    d_int = np.floor(delay).astype(int)
    f = delay - d_int  # fractional part in [0, 1)

    max_d = int(d_int.max())
    min_d = int(d_int.min())
    pad_left = max_d + 2
    ix = np.arange(N) + pad_left

    if order == 3:
        # 4-point stencil: signal[n-D+1], [n-D], [n-D-1], [n-D-2]
        # Lagrange weights at x = -f for nodes at positions 1, 0, -1, -2
        pad_right = max(0, 1 - min_d)
        padded = np.concatenate([np.zeros(pad_left), signal, np.zeros(pad_right)])
        w0 = -f * (1.0 - f) * (2.0 - f) / 6.0
        w1 = (1.0 + f) * (1.0 - f) * (2.0 - f) / 2.0
        w2 = f * (1.0 + f) * (2.0 - f) / 2.0
        w3 = -f * (1.0 + f) * (1.0 - f) / 6.0
        return (
            w0 * padded[ix - d_int + 1]
            + w1 * padded[ix - d_int]
            + w2 * padded[ix - d_int - 1]
            + w3 * padded[ix - d_int - 2]
        )
    else:  # order == 4
        # 5-point stencil: signal[n-D+2] … [n-D-2]
        # Lagrange weights at x = -f for nodes at positions 2, 1, 0, -1, -2
        pad_right = max(0, 2 - min_d)
        padded = np.concatenate([np.zeros(pad_left), signal, np.zeros(pad_right)])
        w0 = (2.0 - f) * (1.0 - f) * f * (1.0 + f) / 24.0
        w1 = -(2.0 - f) * (1.0 - f) * f * (2.0 + f) / 6.0
        w2 = (2.0 - f) * (1.0 - f) * (1.0 + f) * (2.0 + f) / 4.0
        w3 = (2.0 - f) * f * (1.0 + f) * (2.0 + f) / 6.0
        w4 = -(1.0 - f) * f * (1.0 + f) * (2.0 + f) / 24.0
        return (
            w0 * padded[ix - d_int + 2]
            + w1 * padded[ix - d_int + 1]
            + w2 * padded[ix - d_int]
            + w3 * padded[ix - d_int - 1]
            + w4 * padded[ix - d_int - 2]
        )
