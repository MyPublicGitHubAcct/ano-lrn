import numpy as np

from python.delay.line import fractional_delay_line


def chorus(
    signal: np.ndarray,
    rate: float = 1.0,
    depth_ms: float = 2.5,
    delay_ms: float = 25.0,
    mix: float = 0.5,
    fs: int = 44100,
) -> np.ndarray:
    """Chorus: mix the dry signal with a sinusoidally modulated delayed copy.

    delay_ms sets the center delay (20–30 ms detunes without audible pitch shift).
    depth_ms is the LFO swing around that center.
    mix=0 returns the dry signal; mix=1 returns the wet copy only.
    """
    n = len(signal)
    t = np.arange(n) / fs
    depth_s = depth_ms * fs / 1000.0
    center_s = delay_ms * fs / 1000.0
    delay = center_s + depth_s * np.sin(2 * np.pi * rate * t)
    wet = fractional_delay_line(signal, delay)
    return (1.0 - mix) * signal + mix * wet


def flanger(
    signal: np.ndarray,
    rate: float = 0.5,
    depth_ms: float = 2.0,
    delay_ms: float = 2.5,
    feedback: float = 0.5,
    mix: float = 0.5,
    fs: int = 44100,
) -> np.ndarray:
    """Flanger: sinusoidally modulated short delay with feedback.

    delay_ms sets the center delay (0.5–5 ms creates sweeping comb notches).
    depth_ms is the LFO swing around the center delay.
    feedback recirculates the delay output back into the delay input, deepening
    the comb notches; |feedback| < 1 for stability.
    mix=0 returns dry; mix=1 returns the wet (delayed+feedback) signal only.

    A sample-by-sample loop is required because the feedback path means each
    output sample feeds back into the delay buffer.
    """
    n = len(signal)
    t = np.arange(n) / fs
    depth_s = depth_ms * fs / 1000.0
    center_s = delay_ms * fs / 1000.0
    delay_curve = center_s + depth_s * np.sin(2 * np.pi * rate * t)
    delay_curve = np.maximum(1.0, delay_curve)  # min 1-sample delay avoids read/write collision

    max_d = int(np.ceil(center_s + depth_s)) + 2
    buf_len = max_d + 2  # circular buffer must exceed max delay
    buf = np.zeros(buf_len)
    wet = np.zeros(n)

    for i in range(n):
        d = delay_curve[i]
        d_int = int(d)
        frac = d - d_int
        # Linear interpolation from the circular buffer
        i0 = (i - d_int) % buf_len
        i1 = (i - d_int - 1) % buf_len
        delayed = buf[i0] * (1.0 - frac) + buf[i1] * frac
        buf[i % buf_len] = signal[i] + feedback * delayed
        wet[i] = delayed

    return (1.0 - mix) * signal + mix * wet
