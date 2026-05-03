import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter


# ── Source-filter ─────────────────────────────────────────────────────────────

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


def formant_filter(
    signal: np.ndarray,
    formant_freqs: list,
    bandwidths: list,
    fs: int = 44100,
) -> np.ndarray:
    """Cascade of 2nd-order resonators at each formant frequency.

    Each resonator boosts energy at formant_freqs[i] with bandwidth
    bandwidths[i] Hz (–3 dB full bandwidth).
    Pole radius: R = exp(-pi * bw / fs).
    """
    out = np.asarray(signal, dtype=float)
    for f0, bw in zip(formant_freqs, bandwidths):
        R = np.exp(-np.pi * max(bw, 1.0) / fs)
        theta = 2 * np.pi * f0 / fs
        b = [1.0 - R ** 2]
        a = [1.0, -2.0 * R * np.cos(theta), R ** 2]
        out = lfilter(b, a, out).astype(float)
    return out
