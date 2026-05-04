import numpy as np

from python.filters._helpers import _biquad


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
