import numpy as np
from scipy.signal import lfilter


def _biquad(signal: np.ndarray, b: list, a: list) -> np.ndarray:
    return lfilter(b, a, signal).astype(float)


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
