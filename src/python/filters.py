import numpy as np
from scipy.signal import lfilter


def _biquad(signal: np.ndarray, b: list, a: list) -> np.ndarray:
    return lfilter(b, a, signal).astype(float)


# ── EQ / parametric ───────────────────────────────────────────────────────────

def _lp_coeffs(cutoff: float, fs: int, Q: float) -> tuple:
    # Audio EQ Cookbook — Low Pass Filter
    w0 = 2 * np.pi * cutoff / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1 + alpha
    b = [(1 - cos_w0) / 2 / a0, (1 - cos_w0) / a0, (1 - cos_w0) / 2 / a0]
    a = [1.0, -2 * cos_w0 / a0, (1 - alpha) / a0]
    return b, a


def _hp_coeffs(cutoff: float, fs: int, Q: float) -> tuple:
    # Audio EQ Cookbook — High Pass Filter
    w0 = 2 * np.pi * cutoff / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1 + alpha
    b = [(1 + cos_w0) / 2 / a0, -(1 + cos_w0) / a0, (1 + cos_w0) / 2 / a0]
    a = [1.0, -2 * cos_w0 / a0, (1 - alpha) / a0]
    return b, a


def _bp_coeffs(cutoff: float, fs: int, Q: float) -> tuple:
    # Audio EQ Cookbook — Band Pass Filter (constant 0 dB peak gain)
    # b0 = alpha (not sin(w0)/2, which would give constant-skirt-gain with peak = Q)
    w0 = 2 * np.pi * cutoff / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1 + alpha
    b = [alpha / a0, 0.0, -alpha / a0]
    a = [1.0, -2 * cos_w0 / a0, (1 - alpha) / a0]
    return b, a


def _notch_coeffs(cutoff: float, fs: int, Q: float) -> tuple:
    # Audio EQ Cookbook — Notch (Band-Reject) Filter
    w0 = 2 * np.pi * cutoff / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1 + alpha
    b = [1.0 / a0, -2 * cos_w0 / a0, 1.0 / a0]
    a = [1.0, -2 * cos_w0 / a0, (1 - alpha) / a0]
    return b, a


def _ap_coeffs(cutoff: float, fs: int, Q: float) -> tuple:
    # Audio EQ Cookbook — All-Pass Filter
    w0 = 2 * np.pi * cutoff / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    a0 = 1 + alpha
    b = [(1 - alpha) / a0, -2 * cos_w0 / a0, 1.0]
    a = [1.0, -2 * cos_w0 / a0, (1 - alpha) / a0]
    return b, a


def lowpass(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    Q: float = 0.707,
) -> np.ndarray:
    return _biquad(signal, *_lp_coeffs(cutoff, fs, Q))


def highpass(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    Q: float = 0.707,
) -> np.ndarray:
    return _biquad(signal, *_hp_coeffs(cutoff, fs, Q))


def bandpass(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    Q: float = 1.0,
) -> np.ndarray:
    return _biquad(signal, *_bp_coeffs(cutoff, fs, Q))


def notch(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    Q: float = 1.0,
) -> np.ndarray:
    return _biquad(signal, *_notch_coeffs(cutoff, fs, Q))


def allpass(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    Q: float = 0.707,
) -> np.ndarray:
    return _biquad(signal, *_ap_coeffs(cutoff, fs, Q))


# ── Shelving ──────────────────────────────────────────────────────────────────

def _ls_coeffs(cutoff: float, fs: int, gain_db: float) -> tuple:
    # Audio EQ Cookbook — Low-Shelf Filter (shelf slope S = 1)
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * cutoff / fs
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / np.sqrt(2)
    sqrtA = np.sqrt(A)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha
    b = [
        A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha) / a0,
        2 * A * ((A - 1) - (A + 1) * cos_w0) / a0,
        A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha) / a0,
    ]
    a = [
        1.0,
        -2 * ((A - 1) + (A + 1) * cos_w0) / a0,
        ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha) / a0,
    ]
    return b, a


def _hs_coeffs(cutoff: float, fs: int, gain_db: float) -> tuple:
    # Audio EQ Cookbook — High-Shelf Filter (shelf slope S = 1)
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * cutoff / fs
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / np.sqrt(2)
    sqrtA = np.sqrt(A)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha
    b = [
        A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha) / a0,
        -2 * A * ((A - 1) + (A + 1) * cos_w0) / a0,
        A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha) / a0,
    ]
    a = [
        1.0,
        2 * ((A - 1) - (A + 1) * cos_w0) / a0,
        ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha) / a0,
    ]
    return b, a


def lowshelf(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    gain_db: float = 6.0,
) -> np.ndarray:
    return _biquad(signal, *_ls_coeffs(cutoff, fs, gain_db))


def highshelf(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    gain_db: float = 6.0,
) -> np.ndarray:
    return _biquad(signal, *_hs_coeffs(cutoff, fs, gain_db))


# ── Utility ───────────────────────────────────────────────────────────────────

def dc_block(
    signal: np.ndarray,
    cutoff: float = 20.0,
    fs: int = 44100,
) -> np.ndarray:
    """First-order DC blocking filter.

    Removes DC offset and sub-sonic content below `cutoff` Hz.
    Uses the bilinear transform of a 1st-order analog high-pass, so the
    −3 dB point lands exactly at `cutoff` Hz.

    Transfer function: H(z) = (1 − z⁻¹) / (1 + k − (1 − k)·z⁻¹)
    where k = tan(π·cutoff/fs).  H(1) = 0 exactly (structural zero at DC).
    """
    k = np.tan(np.pi * cutoff / fs)
    b = [1.0 / (1.0 + k), -1.0 / (1.0 + k)]
    a = [1.0, -(1.0 - k) / (1.0 + k)]
    return lfilter(b, a, signal).astype(float)
