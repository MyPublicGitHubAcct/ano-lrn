import numpy as np

from python.delay.line import fractional_delay_line


def ping_pong_delay(
    signal: np.ndarray,
    fs: int,
    delay_time: float,
    feedback: float = 0.5,
    mix: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ping-pong stereo delay.

    Echoes alternate between right and left channels: the first echo appears in
    the right channel at delay_time seconds; the second in the left channel at
    2*delay_time with amplitude feedback; the third in right at 3*delay_time
    with amplitude feedback**2; and so on.

    Two cross-fed fractional delay lines implement this: each iteration passes
    the current echo through another fractional delay of delay_time*fs samples,
    alternating the output channel and attenuating by feedback.

    signal:     mono input array
    fs:         sample rate in Hz
    delay_time: echo period in seconds
    feedback:   gain per bounce, [0, 1); 0 = single echo, 0.9 = many echoes
    mix:        0 = fully dry (both channels), 1 = fully wet

    Returns (t, L, R) — time axis and left/right output arrays, each the same
    length as signal.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    t = np.arange(N) / fs
    D = delay_time * fs  # delay in samples

    L_wet = np.zeros(N)
    R_wet = np.zeros(N)

    current = signal.copy()
    for k in range(500):
        current = fractional_delay_line(current, D)
        if k % 2 == 0:
            R_wet += current
        else:
            L_wet += current
        current = current * feedback
        if np.max(np.abs(current)) < 1e-8:
            break

    L = (1.0 - mix) * signal + mix * L_wet
    R = (1.0 - mix) * signal + mix * R_wet
    return t, L, R
