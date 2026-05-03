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


# ── ZDF Ladder ───────────────────────────────────────────────────────────────

def moog_ladder(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    resonance: float = 0.0,
) -> np.ndarray:
    """Zero-delay feedback 4-pole ladder filter (Moog/Minimoog emulation).

    Zavalishin TPT topology: four 1-pole ZDF lowpass stages in series with
    global resonance feedback. resonance in [0, 1] maps to feedback k in
    [0, 4]; k = 4 is the self-oscillation threshold. DC gain is 1/(1 + k).
    """
    g = np.tan(np.pi * cutoff / fs)
    k = resonance * 4.0
    G = g / (1.0 + g)
    inv_1pg = 1.0 / (1.0 + g)

    G2 = G * G
    G3 = G2 * G
    G4 = G3 * G
    denom = 1.0 + k * G4

    s1 = s2 = s3 = s4 = 0.0
    out = np.empty(len(signal))

    for n, x in enumerate(signal):
        # State contributions (normalised for current g)
        S1 = s1 * inv_1pg
        S2 = s2 * inv_1pg
        S3 = s3 * inv_1pg
        S4 = s4 * inv_1pg

        # Solve for y4 without unit delay in the feedback path
        y4 = (G4 * x + G3 * S1 + G2 * S2 + G * S3 + S4) / denom

        # Propagate through stages and update states
        u1 = x - k * y4
        v1 = (u1 - s1) * G;  y1 = v1 + s1;  s1 = y1 + v1
        v2 = (y1 - s2) * G;  y2 = v2 + s2;  s2 = y2 + v2
        v3 = (y2 - s3) * G;  y3 = v3 + s3;  s3 = y3 + v3
        v4 = (y3 - s4) * G;  y4 = v4 + s4;  s4 = y4 + v4

        out[n] = y4

    return out


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
