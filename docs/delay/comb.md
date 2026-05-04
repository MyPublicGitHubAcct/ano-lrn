# Comb Filters

## `feedback_delay`

An IIR feedback comb filter:

```text
y[n] = x[n] + feedback * y[n − D]
```

Transfer function:

```text
H(z) = 1 / (1 − feedback · z^(−D))
```

The impulse response is a sequence of exponentially decaying echoes at lags D, 2D, 3D, … with amplitudes 1, `feedback`, `feedback²`, … Stability requires `|feedback| < 1`. At `feedback = 0` the system reduces to H(z) = 1 (identity).

**Frequency response:** resonant peaks at frequencies f = k · fs / D for integer k, with sharpness controlled by `feedback` (closer to 1 → narrower peaks).

**Use cases:** simple echo and reverb topologies.

---

## `comb_filter`

A FIR feedforward comb filter:

```text
y[n] = x[n] + gain * x[n − D]
```

Transfer function:

```text
H(z) = 1 + gain · z^(−D)
```

Creates alternating peaks and notches spaced by fs / D Hz. Unlike the IIR variant, this filter is always stable regardless of `gain`.

**Peak/notch spacing:** notches occur at f_notch = (2k+1) · fs / (2D) for k = 0, 1, 2, …

**Use cases:** chorus, flanger, and acoustic room simulation.
