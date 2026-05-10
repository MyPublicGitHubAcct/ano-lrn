import numpy as np

from python.delay.line import fractional_delay_line
from python.nonlinear.clipping import soft_clip


def tape_delay(
    signal: np.ndarray,
    fs: int,
    delay_time: float = 0.2,
    feedback: float = 0.4,
    flutter_rate: float = 10.0,
    flutter_depth: float = 0.0005,
    wow_rate: float = 0.5,
    wow_depth: float = 0.002,
    saturation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tape echo / tape delay emulator.

    Models a magnetic tape loop between record and playback heads.  The
    primary delay is a fractional-delay read from the signal; wow and flutter
    add sinusoidal perturbations to that read position.

    signal:        mono input array
    fs:            sample rate in Hz
    delay_time:    nominal tape-loop delay in seconds
    feedback:      recirculation gain, [0, 1); 0 = single echo, 0.9 = long tail
    flutter_rate:  fast tape-speed variation frequency in Hz (typically 8–12 Hz)
    flutter_depth: flutter depth in seconds (peak read-position swing)
    wow_rate:      slow tape-speed variation frequency in Hz (typically < 2 Hz)
    wow_depth:     wow depth in seconds (peak read-position swing)
    saturation:    soft-clipping drive in the feedback path; 0 = off, 1 = heavy

    Returns (t, output) where output is the wet echo trail (no dry signal).
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    t = np.arange(N) / fs

    D = delay_time * fs  # nominal delay in samples

    n_axis = np.arange(N)
    wow = wow_depth * fs * np.sin(2 * np.pi * wow_rate * n_axis / fs)
    flutter = flutter_depth * fs * np.sin(2 * np.pi * flutter_rate * n_axis / fs)
    delay_curve = np.maximum(1.0, D + wow + flutter)

    output = np.zeros(N)
    current = signal.copy()
    for _ in range(500):
        current = fractional_delay_line(current, delay_curve)
        output += current
        if saturation > 0.0:
            current = soft_clip(current, drive=1.0 + saturation * 4.0)
        current = current * feedback
        if np.max(np.abs(current)) < 1e-8:
            break

    return t, output
