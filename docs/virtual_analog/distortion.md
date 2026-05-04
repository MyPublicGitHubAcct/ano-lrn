# Virtual Analog Distortion

## `diode_clip`

Models asymmetric clipping characteristic of a diode pair:

- **Positive half:** hard clip at `+threshold` (ideal diode forward conduction).
- **Negative half:** soft clip via tanh, modelling the diode's gradual turn-on region.

```text
y[n] = threshold                              if x[n] > threshold
y[n] = x[n]                                  if 0 ≤ x[n] ≤ threshold
y[n] = −threshold · tanh(−x[n] / threshold)  if x[n] < 0
```

The asymmetry generates both even and odd harmonics, giving a character similar to BJT transistor or diode-based guitar overdrive circuits.

---

## `analog_saturate`

Third-order polynomial approximation to a triode vacuum tube transfer curve:

```text
x_d = clip(x[n] · drive, −1, 1)
y[n] = x_d − x_d³ / 3
```

The input is first driven by `drive` and then clamped to avoid the polynomial growing unbounded. The cubic term introduces only odd harmonics (3rd harmonic dominant). The derivative dy/dx = 1 − x² ≥ 0 everywhere on [−1, 1], so the transfer function is monotone (no foldback distortion).

**Comparison to tanh:** the polynomial model saturates more abruptly than tanh, matching the sharp knee of a triode running into grid-current clipping.
