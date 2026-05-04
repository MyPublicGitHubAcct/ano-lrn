import numpy as np


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
