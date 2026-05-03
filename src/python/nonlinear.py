import numpy as np


# ── Nonlinear ─────────────────────────────────────────────────────────────────

def hard_clip(signal: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Clip signal symmetrically to [-threshold, threshold]."""
    return np.clip(signal, -threshold, threshold)


def soft_clip(signal: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh saturation; output stays in (-1, 1) regardless of drive."""
    return np.tanh(signal * drive)


def waveshape(signal: np.ndarray, coeffs: list) -> np.ndarray:
    """Polynomial waveshaping: y = c0 + c1*x + c2*x^2 + ...

    coeffs[i] is the coefficient of x^i (ascending power order).
    """
    return np.polynomial.polynomial.polyval(np.asarray(signal, dtype=float), coeffs)


def bitcrush(signal: np.ndarray, bits: int = 8) -> np.ndarray:
    """Reduce signal to `bits`-bit amplitude resolution.

    Quantises to 2^bits evenly spaced levels spanning [-1, 1].
    """
    levels = 2 ** bits
    half = levels / 2.0
    return np.round(signal * half) / half
