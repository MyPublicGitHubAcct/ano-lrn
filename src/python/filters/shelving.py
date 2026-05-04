import numpy as np

from python.filters._helpers import _biquad


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
