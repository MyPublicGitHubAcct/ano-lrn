import numpy as np


def pitch_shift(
    signal: np.ndarray,
    fs: int,
    semitones: float,
) -> np.ndarray:
    """Time-domain pitch shifter via two Hann-windowed overlapping read heads.

    Uses a single circular delay buffer of length B (next power of 2 above
    50 ms at `fs`).  Two read heads traverse the buffer at pitch ratio
    r = 2^(semitones/12); their grain pointers are staggered by B/2 so the
    two Hann envelopes always sum to 1, providing seamless crossfades when a
    head wraps.

    The grain pointer p represents the delay (age in samples) of the sample
    being read.  It advances at (1 − r) per step: negative for pitch-up
    (delay shrinks, read head catches write head), positive for pitch-down
    (delay grows, read head falls behind).  When p wraps at the buffer
    boundary the crossfade absorbs the discontinuity.

    signal:   mono input array
    fs:       sample rate in Hz
    semitones: shift amount; positive = up, negative = down

    Returns an array the same length as `signal`.  The first B//2 output
    samples are silent (startup latency); for semitones=0, out[B//2:] equals
    signal[:N-B//2] to floating-point precision.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    r = 2.0 ** (semitones / 12.0)

    # Next power of 2 >= 50 ms — long enough for the crossfade to cover one
    # full period at 20 Hz while keeping latency under 100 ms.
    B = 1
    while B < int(fs * 0.05):
        B <<= 1

    n_axis = np.arange(N, dtype=float)

    # p0 ∈ [0, B): delay of the sample read by head 0.
    # Starting at B/2 places head 0 at Hann maximum on the first output sample.
    # Head 1 is always B/2 ahead in phase so the two Hann windows sum to 1.
    p0 = (B / 2.0 + (1.0 - r) * n_axis) % B
    p1 = (p0 + B / 2.0) % B

    win0 = 0.5 * (1.0 - np.cos(2.0 * np.pi * p0 / B))
    win1 = 0.5 * (1.0 - np.cos(2.0 * np.pi * p1 / B))

    # Source positions in the original signal (fractional; may be negative
    # during the B/2-sample startup window — zero-padding handles those).
    src0 = n_axis - p0
    src1 = n_axis - p1

    # Pad: B zeros on the left absorb negative indices; 1 zero on the right
    # is the interpolation guard for the last sample.
    padded = np.concatenate([np.zeros(B), signal, np.zeros(1)])

    i0 = np.floor(src0 + B).astype(int)
    f0 = src0 + B - i0
    i1 = np.floor(src1 + B).astype(int)
    f1 = src1 + B - i1

    top = len(padded) - 2  # highest safe base index for linear interpolation
    i0 = np.clip(i0, 0, top)
    i1 = np.clip(i1, 0, top)

    s0 = padded[i0] * (1.0 - f0) + padded[i0 + 1] * f0
    s1 = padded[i1] * (1.0 - f1) + padded[i1 + 1] * f1

    return win0 * s0 + win1 * s1
