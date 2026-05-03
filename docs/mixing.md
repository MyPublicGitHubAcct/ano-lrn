# Mixing

`src/python/mixing.py` implements gain, summation, crossfade, and normalisation — the essential operations for routing and level control.

## Summary

| Function | Key parameters | Description |
| --- | --- | --- |
| `gain` | `signal`, `gain_db` | Scale amplitude by dB amount |
| `mix` | `signals`, `weights` | Weighted sum of multiple signals |
| `crossfade` | `signal_a`, `signal_b`, `position` | Linear blend between two signals |
| `normalize` | `signal`, `target_db` | Scale peak to target dB level |

---

### `gain`

Applies a gain in decibels:

```text
y[n] = x[n] · 10^(gain_db / 20)
```

Positive `gain_db` amplifies; negative attenuates. `gain_db = 0` is unity gain (identity). At `gain_db = +6` the amplitude approximately doubles; at `−6` it halves.

---

### `mix`

Computes a weighted sum of N signals:

```text
y[n] = Σᵢ  wᵢ · xᵢ[n]
```

If `weights` is `None`, all signals are averaged equally (`wᵢ = 1/N`). Weighted sums are the core of any mixer bus. To prevent clipping when summing N equal-level signals at unity weight, scale weights to `1/N` or reduce individual track levels by `−20·log10(N)` dB before mixing.

---

### `crossfade`

Linear crossfade between two signals:

```text
y[n] = (1 − p) · a[n] + p · b[n]        p ∈ [0, 1]
```

At `position = 0` the output is fully `signal_a`; at `1` it is fully `signal_b`. The summed power at the crossover point is `(0.5a + 0.5b)`, which can cause a 6 dB power dip compared to the original signals if they are coherent. Use equal-power crossfade (`cos/sin`) when a smooth loudness transition is required.

---

### `normalize`

Scales the signal so that its peak absolute value matches `target_db`:

```text
peak   = max|x[n]|
y[n]   = x[n] · 10^(target_db / 20) / peak
```

Silent signals (peak < 1e−12) are returned unchanged to avoid division by near-zero. Common targets: 0 dB (full scale), −3 dB (3 dB headroom), −12 dB (broadcast safety margin).
