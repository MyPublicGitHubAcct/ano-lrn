# Stereo

## `pan`

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

## `stereo_widen`

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

## Parameter Ranges

**`pan`**

| Parameter | Range | Notes |
| --- | --- | --- |
| `position` | [−1, 1] | −1 = hard left (L=1, R=0); 0 = centre (L=R=1/√2); +1 = hard right (L=0, R=1) |

**`stereo_widen`**

| Parameter | Range | Notes |
| --- | --- | --- |
| `width` | ≥ 0 | `0` = mono (side zeroed); `1` = unchanged; `> 1` widens; large values may cause destructive anti-phase cancellation on mono playback |
