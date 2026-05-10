import numpy as np

from python.delay.line import fractional_delay_line


def multitap_delay(
    signal: np.ndarray,
    fs: int,
    taps: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multitap fractional delay line with stereo panning.

    Each entry in taps contributes an independently delayed and panned copy of
    the input signal to the stereo output.  All taps share the same input array;
    each one calls fractional_delay_line with a scalar delay so there is no
    per-sample Python loop.

    taps: list of (delay_seconds, gain, pan) tuples
        delay_seconds: tap position in seconds (fractional sample accuracy via
                       Lagrange interpolation)
        gain:          linear amplitude scaling for this tap
        pan:           stereo position in [-1, 1]; -1 = full left, 0 = centre,
                       +1 = full right; resolved with constant-power (cos/sin)
                       panning: theta = (pan+1)*pi/4

    Returns (t, L, R) — time axis and stereo output arrays, each the same
    length as signal.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    t = np.arange(N) / fs

    L_out = np.zeros(N)
    R_out = np.zeros(N)

    for delay_seconds, gain, pan in taps:
        D = delay_seconds * fs
        tap_signal = fractional_delay_line(signal, D)
        theta = (pan + 1.0) * (np.pi / 4.0)  # [-1,1] → [0, π/2]
        L_out += gain * np.cos(theta) * tap_signal
        R_out += gain * np.sin(theta) * tap_signal

    return t, L_out, R_out
