# Virtual Analog

`src/python/virtual_analog.py` models nonlinear and analog-circuit characteristics in the digital domain.

## Summary

| Function | Key parameters | Description |
| --- | --- | --- |
| `moog_ladder` | `cutoff`, `fs`, `resonance` | Discretised 4-pole Moog ladder filter |
| `diode_clip` | `threshold` | Asymmetric hard/soft diode clipping |
| `analog_saturate` | `drive` | 3rd-order polynomial tube saturation |

---

### `moog_ladder`

The Moog transistor ladder filter is a 4-pole (24 dB/octave) lowpass with nonlinear resonance. The Huovilainen discretisation cascades four one-pole stages with tanh nonlinearities and a feedback path:

```text
g = 1 − exp(−2π · fc)           (one-pole coefficient)
k = 4 · resonance                (feedback gain)

x_fb   = x[n] − k · s₃         (input minus resonance feedback)
s₀    += g · (tanh(x_fb) − tanh(s₀))
s₁    += g · (tanh(s₀)  − tanh(s₁))
s₂    += g · (tanh(s₁)  − tanh(s₂))
s₃    += g · (tanh(s₂)  − tanh(s₃))
y[n]   = s₃
```

At `resonance = 0` the filter is a clean 4th-order lowpass. As resonance approaches 1 the filter self-oscillates at `cutoff`, producing a pure sinusoid. The tanh in each stage limits the resonance amplitude.

**Cutoff tracking:** the filter does not track pitch accurately at high frequencies due to the bilinear approximation; pre-warping (`fc = sin(π·f/fs) / π`) improves accuracy at the cost of a nonlinear cutoff mapping.

---

### `diode_clip`

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

### `analog_saturate`

Third-order polynomial approximation to a triode vacuum tube transfer curve:

```text
x_d = clip(x[n] · drive, −1, 1)
y[n] = x_d − x_d³ / 3
```

The input is first driven by `drive` and then clamped to avoid the polynomial growing unbounded. The cubic term introduces only odd harmonics (3rd harmonic dominant). The derivative dy/dx = 1 − x² ≥ 0 everywhere on [−1, 1], so the transfer function is monotone (no foldback distortion).

**Comparison to tanh:** the polynomial model saturates more abruptly than tanh, matching the sharp knee of a triode running into grid-current clipping.
