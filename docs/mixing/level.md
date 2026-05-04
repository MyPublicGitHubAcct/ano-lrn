# Level

## `gain`

Applies a gain in decibels:

```text
y[n] = x[n] · 10^(gain_db / 20)
```

Positive `gain_db` amplifies; negative attenuates. `gain_db = 0` is unity gain (identity). At `gain_db = +6` the amplitude approximately doubles; at `−6` it halves.

---

## `normalize`

Scales the signal so that its peak absolute value matches `target_db`:

```text
peak   = max|x[n]|
y[n]   = x[n] · 10^(target_db / 20) / peak
```

Silent signals (peak < 1e−12) are returned unchanged to avoid division by near-zero. Common targets: 0 dB (full scale), −3 dB (3 dB headroom), −12 dB (broadcast safety margin).
