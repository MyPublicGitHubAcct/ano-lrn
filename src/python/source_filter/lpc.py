import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter


def lpc_coeffs(signal: np.ndarray, order: int = 12) -> np.ndarray:
    """Linear predictive coding (LPC) analysis via autocorrelation method.

    Returns an array of `order` predictor coefficients a[1..p] such that
    the all-pole filter H(z) = 1 / (1 - sum_k a[k] z^-k) models the spectral
    envelope of `signal`.
    """
    r = np.correlate(signal, signal, mode="full")
    r = r[len(r) // 2 :]  # non-negative lags
    R = toeplitz(r[:order])
    return np.linalg.solve(R, r[1 : order + 1])


def lpc_synthesize(excitation: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply an all-pole LPC synthesis filter to an excitation signal.

    coeffs: predictor coefficients as returned by lpc_coeffs().
    Transfer function: H(z) = 1 / A(z)
    where A(z) = 1 - coeffs[0]*z^-1 - coeffs[1]*z^-2 - ...
    """
    a = np.concatenate([[1.0], -np.asarray(coeffs, dtype=float)])
    return lfilter([1.0], a, excitation).astype(float)
