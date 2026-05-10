# Multitap Fractional Delay Line

A multitap delay reads independently delayed and gain-scaled copies of a mono input signal from a shared buffer and places each copy at an arbitrary stereo position using a constant-power panning law, producing a `(t, L, R)` stereo output.

---

## `multitap_delay`

```python
multitap_delay(signal, fs, taps) -> (t, L, R)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `signal` | `ndarray` | Mono input array |
| `fs` | `int` | Sample rate in Hz |
| `taps` | `list[tuple]` | List of `(delay_seconds, gain, pan)` tuples |

Each tap tuple:

| Field | Type | Description |
| --- | --- | --- |
| `delay_seconds` | `float` | Tap delay in seconds (fractional sample accuracy) |
| `gain` | `float` | Linear amplitude scaling for this tap |
| `pan` | `float` | Stereo position in `[−1, 1]`; `−1` = full left, `0` = centre, `+1` = full right |

Returns `(t, L, R)` — a time axis and left/right output arrays, all the same length as `signal`.

---

## Signal Flow

All taps share the original input signal; each tap applies one call to `fractional_delay_line` with its scalar `delay_seconds × fs` delay, then routes the result through a constant-power pan into L and R:

```text
signal ──┬──→ [Delay D₁] × gain₁ ─→ [Pan θ₁] ─→ L += cos(θ₁) · tap₁
         │                                         R += sin(θ₁) · tap₁
         ├──→ [Delay D₂] × gain₂ ─→ [Pan θ₂] ─→ L += cos(θ₂) · tap₂
         │                                         R += sin(θ₂) · tap₂
         └──→  …
```

There is no feedback and no inter-tap interaction. The output is the superposition of all tap contributions.

---

## Constant-Power Panning

The `pan` value is mapped to an angle `θ` in `[0, π/2]` and resolved via cosine/sine:

```
θ  = (pan + 1) × π/4          # pan=-1 → θ=0, pan=0 → θ=π/4, pan=+1 → θ=π/2
Lᵢ = gain × cos(θ) × tap_signal
Rᵢ = gain × sin(θ) × tap_signal
```

This preserves constant total power at all pan positions:

```
Lᵢ² + Rᵢ² = gain² × (cos²θ + sin²θ) = gain²
```

| `pan` | `θ` | L gain | R gain |
| --- | --- | --- | --- |
| −1 (full left) | 0 | 1 | 0 |
| 0 (centre) | π/4 | 1/√2 ≈ 0.707 | 1/√2 ≈ 0.707 |
| +1 (full right) | π/2 | 0 | 1 |

---

## Fractional Delay

Each tap's delay is resolved by `fractional_delay_line` using 4-point Lagrange polynomial interpolation (default order 3). This allows `delay_seconds × fs` to be a non-integer number of samples while preserving sub-sample timing accuracy — essential when `delay_seconds × fs` is not an integer (virtually always the case for musically meaningful delay times).

---

## Implementation

The implementation is fully vectorised over samples. The only Python loop is over taps (typically 2–16 entries), not over audio samples:

```python
for delay_seconds, gain, pan in taps:
    D = delay_seconds * fs
    tap_signal = fractional_delay_line(signal, D)
    theta = (pan + 1.0) * (pi / 4.0)
    L_out += gain * cos(theta) * tap_signal
    R_out += gain * sin(theta) * tap_signal
```

---

## Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `delay_seconds` | 0 – signal length / fs | 0 = no delay (tap reproduced at t=0); avoid delays longer than the signal |
| `gain` | 0 – 1 (or > 1 for boost) | Gains summing to > 1 across taps can cause clipping |
| `pan` | [−1, 1] | −1 = full left, 0 = centre, +1 = full right |
| Number of taps | 1 – ~16 | Each tap is one `fractional_delay_line` call; cost scales linearly |

---

## VCV Rack Notes

In a C++ port:

- Maintain a single circular ring buffer of length `max(delay_seconds) × fs + guard` samples.
- For each tap, compute the read pointer as `write_ptr − delay_samples` (with fractional part driving linear or Lagrange interpolation).
- Apply the same `cos/sin` gain factors to route each read to the L and R output buses.
- The tap parameters (`delay`, `gain`, `pan`) can be CV-modulated per sample; update the read pointer offset each tick.
