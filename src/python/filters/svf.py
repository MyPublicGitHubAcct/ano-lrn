import numpy as np


def svf(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    resonance: float = 0.0,
    mode: str = "lp",
) -> "np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Chamberlin state-variable filter: simultaneous LP/BP/HP outputs.

    Two integrator stages in a feedback loop with a resonance path, solved
    sample-by-sample (Zölzer/Chamberlin topology). Because f and q are simple
    scalar multipliers in the loop, cutoff and resonance can be changed each
    sample without instability.

    resonance in [0, 1): 0 → Butterworth (maximally flat, q = √2),
    approaching 1 → self-oscillation (q → 0).
    mode: 'lp', 'bp', 'hp', 'notch', or 'all' (returns (lp, bp, hp) tuple).
    """
    q = np.sqrt(2.0) * (1.0 - resonance)
    # Stability requires f(f + 2q) < 4; clamp f to the positive root.
    f_max = np.sqrt(q * q + 4.0) - q
    f = min(2.0 * np.sin(np.pi * cutoff / fs), f_max * (1.0 - 1e-9))

    n = len(signal)
    lp_buf = np.empty(n)
    bp_buf = np.empty(n)
    hp_buf = np.empty(n)

    lp_s = bp_s = 0.0
    for i, x in enumerate(signal):
        lp_s = lp_s + f * bp_s
        hp_s = x - lp_s - q * bp_s
        bp_s = f * hp_s + bp_s
        lp_buf[i] = lp_s
        bp_buf[i] = bp_s
        hp_buf[i] = hp_s

    if mode == "lp":
        return lp_buf
    if mode == "bp":
        return bp_buf
    if mode == "hp":
        return hp_buf
    if mode == "notch":
        return lp_buf + hp_buf
    if mode == "all":
        return lp_buf, bp_buf, hp_buf
    raise ValueError(f"Unknown mode {mode!r}; expected 'lp', 'bp', 'hp', 'notch', or 'all'")


def zdf_svf(
    signal: np.ndarray,
    cutoff: float,
    fs: int = 44100,
    resonance: float = 1.0 / np.sqrt(2),
    mode: str = "lp",
) -> "np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Zavalishin TPT zero-delay-feedback SVF: exact −3 dB at cutoff for all fc/fs.

    Uses bilinear (trapezoidal) integrators solved by simultaneous substitution,
    eliminating the one-sample feedback delay present in the Chamberlin SVF.
    g = tan(π·cutoff/fs) pre-warps the digital cutoff so the −3 dB point is
    exactly `cutoff` Hz, stable and accurate up to fs/2.

    resonance is the Q factor: 1/√2 ≈ 0.707 → Butterworth (default); higher
    values produce a resonant peak; Q → ∞ approaches self-oscillation.
    mode: 'lp', 'bp', 'hp', 'notch', or 'all' (returns (lp, bp, hp) tuple).
    """
    g = np.tan(np.pi * cutoff / fs)
    k = 1.0 / resonance          # damping k = 1/Q; k = √2 for Butterworth
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2

    n = len(signal)
    lp_buf = np.empty(n)
    bp_buf = np.empty(n)
    hp_buf = np.empty(n)

    s1 = s2 = 0.0
    for i, x in enumerate(signal):
        v3 = x - s2
        v1 = a1 * s1 + a2 * v3
        v2 = s2 + a2 * s1 + a3 * v3
        s1 = 2.0 * v1 - s1
        s2 = 2.0 * v2 - s2
        lp_buf[i] = v2
        bp_buf[i] = v1
        hp_buf[i] = x - k * v1 - v2

    if mode == "lp":
        return lp_buf
    if mode == "bp":
        return bp_buf
    if mode == "hp":
        return hp_buf
    if mode == "notch":
        return lp_buf + hp_buf
    if mode == "all":
        return lp_buf, bp_buf, hp_buf
    raise ValueError(f"Unknown mode {mode!r}; expected 'lp', 'bp', 'hp', 'notch', or 'all'")
