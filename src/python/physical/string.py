import numpy as np

from python.generators._helpers import _time_axis


def pluck(
    freq: float,
    fs: int,
    duration: float,
    damping: float = 0.99,
    pickup: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Karplus-Strong plucked string synthesis.

    Args:
        freq:     Fundamental frequency in Hz.
        fs:       Sample rate in Hz.
        duration: Output length in seconds.
        damping:  Loss filter coefficient (0–1); lower values decay faster.
        pickup:   Read position as a fraction of the delay line length (0–1).
                  At 0.5 the pickup sits at the string midpoint, cancelling
                  even harmonics like a guitar neck vs. bridge pickup.
        seed:     RNG seed for reproducible excitation noise.

    Returns:
        (t, signal) — time axis and synthesised pluck waveform.
    """
    t = _time_axis(fs, duration)
    N = len(t)

    # Delay line length sets the fundamental period.
    # The averaging filter advances the effective center delay by 0.5 samples,
    # so add 0.5 before rounding to keep the tuning accurate.
    D = max(2, int(round(fs / freq + 0.5)))

    # Excite with a burst of white noise filling the delay line
    rng = np.random.default_rng(seed)
    buf = rng.standard_normal(D)

    # Pickup offset: how many samples behind the write pointer we read
    pickup_offset = int(round(pickup * D)) % D

    out = np.zeros(N)
    ptr = 0  # write pointer into the ring buffer

    for n in range(N):
        # Averaging loss filter: buf[ptr] = y[n-D] (oldest), buf[(ptr+1)%D] = y[n-D+1]
        # A ring buffer of length D holds y[n-D] through y[n-1]; the only adjacent
        # pair both available without extending the buffer is (y[n-D], y[n-D+1]).
        new_val = damping * 0.5 * (buf[ptr] + buf[(ptr + 1) % D])
        buf[ptr] = new_val
        # Pickup: read at pickup_offset steps ahead of the oldest position (bridge end)
        out[n] = buf[(ptr + pickup_offset) % D]
        ptr = (ptr + 1) % D

    return t, out
