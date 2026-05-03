# Delay

`src/python/delay.py` implements three delay-based building blocks: a pure sample delay, an IIR feedback comb, and a FIR feedforward comb.

## Summary

| Function | Key parameters | Description |
| --- | --- | --- |
| `delay_line` | `delay_samples` | Pure integer-sample delay; shifts signal right by D samples |
| `feedback_delay` | `delay_samples`, `feedback` | IIR comb filter with exponential echo decay |
| `comb_filter` | `delay_samples`, `gain` | FIR feedforward comb; peaks and notches spaced by fs/D Hz |

---

### `delay_line`

A causal integer-sample delay implemented as:

```text
y[n] = x[n - D]
```

Transfer function: H(z) = z^(−D). All samples before index D are zero (initial rest). This is the simplest possible DSP operation and serves as the building block for the two comb filters below. It is also used by `spatial.haas` and `spatial.pan`.

---

### `feedback_delay`

An IIR feedback comb filter:

```text
y[n] = x[n] + feedback * y[n - D]
```

Transfer function:

```text
H(z) = 1 / (1 − feedback · z^(−D))
```

The impulse response is a sequence of exponentially decaying echoes at lags D, 2D, 3D, … with amplitudes 1, `feedback`, `feedback²`, … Stability requires `|feedback| < 1`. At `feedback = 0` the system reduces to H(z) = 1 (identity). Used in simple echo and reverb topologies.

**Frequency response:** the transfer function creates resonant peaks (standing waves) where the denominator is small. Peaks are at frequencies f = k · fs / D for integer k, with sharpness controlled by `feedback` (closer to 1 → narrower peaks).

---

### `comb_filter`

A FIR feedforward comb filter:

```text
y[n] = x[n] + gain * x[n - D]
```

Transfer function:

```text
H(z) = 1 + gain · z^(−D)
```

Creates alternating peaks and notches spaced by fs / D Hz. The magnitude at frequency f is |1 + gain · e^(−j2πfD/fs)|. Unlike the IIR variant, this filter is always stable regardless of `gain`. Used for chorus, flanger, and acoustic room simulation.

**Peak/notch spacing:** notches occur where the delay adds a half-period phase shift: f_notch = (2k+1) · fs / (2D) for k = 0, 1, 2, …
