# Blend

## `mix`

Computes a weighted sum of N signals:

```text
y[n] = Σᵢ  wᵢ · xᵢ[n]
```

If `weights` is `None`, all signals are averaged equally (`wᵢ = 1/N`). Weighted sums are the core of any mixer bus. To prevent clipping when summing N equal-level signals at unity weight, scale weights to `1/N` or reduce individual track levels by `−20·log10(N)` dB before mixing.

---

## `crossfade`

Linear crossfade between two signals:

```text
y[n] = (1 − p) · a[n] + p · b[n]        p ∈ [0, 1]
```

At `position = 0` the output is fully `signal_a`; at `1` it is fully `signal_b`. The summed power at the crossover point is `(0.5a + 0.5b)`, which can cause a 6 dB power dip compared to the original signals if they are coherent. Use equal-power crossfade (`cos/sin`) when a smooth loudness transition is required.
