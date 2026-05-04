# Sweep Generator

A frequency sweep that moves continuously through a range of frequencies over time.

| Generator | Key parameters | Description |
| --- | --- | --- |
| `generate_chirp` | `f_start`, `f_end`, `method` | Frequency sweep; logarithmic (default) or linear |

Returns `(t, signal)` — a time axis and signal array sampled at `fs` Hz over `duration` seconds.

---

## `generate_chirp`

A frequency sweep from `f_start` to `f_end` over `duration` seconds. Two methods are available:

**Logarithmic (default)** — instantaneous frequency increases exponentially:

```text
f(t) = f_start · e^(k·t),   k = ln(f_end / f_start) / T
phase(t) = 2π f_start (e^(k·t) − 1) / k
```

Each octave takes the same wall-clock time, matching the logarithmic spacing of the musical scale. Preferred for audio-band frequency response measurements.

**Linear** — instantaneous frequency increases linearly:

```text
f(t) = f_start + (f_end − f_start) · t / T
phase(t) = 2π (f_start · t + (f_end − f_start) · t² / 2T)
```

Equal Hz per second; concentrates dwell time at low frequencies relative to linear frequency spacing.

**Use cases:** sweeping a filter's passband, visualizing time-frequency behavior via spectrogram.
