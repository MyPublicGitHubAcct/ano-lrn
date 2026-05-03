# Spatial

`src/python/spatial.py` implements three stereo panning and spatial width functions.

## Summary

| Function | Key parameters | Returns | Description |
| --- | --- | --- | --- |
| `pan` | `position` | `(left, right)` | Equal-power stereo panning of a mono signal |
| `stereo_widen` | `width` | `(left, right)` | M/S stereo width control |
| `haas` | `delay_samples` | `(left, right)` | Haas precedence effect via inter-channel delay |

---

### `pan`

Equal-power (constant-power) panning maps a mono signal to a stereo pair:

```text
θ = (position + 1) · π/4         ∈ [0, π/2]
L[n] = x[n] · cos(θ)
R[n] = x[n] · sin(θ)
```

The law L² + R² = x² holds at every position, so the total power is conserved regardless of pan position. This matches the perceived loudness law better than linear crossfade for audio panning:

| `position` | L gain | R gain |
| --- | --- | --- |
| −1 (hard left) | 1.0 | 0.0 |
| 0 (centre) | 0.707 | 0.707 |
| +1 (hard right) | 0.0 | 1.0 |

---

### `stereo_widen`

Mid/side (M/S) stereo processing applies a width multiplier to the side component:

```text
M = (L + R) / 2        (mono sum)
S = (L − R) / 2        (difference / stereo width)
L' = M + S · width
R' = M − S · width
```

At `width = 0` the output is mono (S = 0). At `width = 1` the output is identical to the input. At `width > 1` the stereo image is widened. The M/S decomposition is fully invertible for any finite width value.

**Use case:** mastering bus processing; creating spatial width from a narrow stereo source.

---

### `haas`

The Haas (precedence) effect exploits the ear's directional fusion mechanism. When the same signal arrives at two ears within about 40 ms of each other, the brain fuses the two arrivals into a single perceived source whose location is biased toward the first arrival:

```text
L[n] = x[n]
R[n] = x[n − D]
```

At delay values D < ~1750 samples (< 40 ms at 44100 Hz) the right channel is heard as part of the same source, creating a sense of width without a distinct echo. Beyond ~40 ms the delay becomes a perceptible echo.
