# EQ / Parametric Filters

Five biquad filters that shape the frequency response by passing, attenuating, or phase-shifting specific bands. All are implemented using the [Audio EQ Cookbook](https://www.musicdsp.org/en/latest/Filters/197-rbj-audio-eq-cookbook.html) coefficient formulas.

| Filter | Key parameters | Passes | Rejects | Q effect |
| --- | --- | --- | --- | --- |
| `lowpass` | `cutoff`, `Q` | Below cutoff | Above cutoff | Higher Q → resonant peak below cutoff |
| `highpass` | `cutoff`, `Q` | Above cutoff | Below cutoff | Higher Q → resonant peak above cutoff |
| `bandpass` | `cutoff`, `Q` | Band around cutoff | DC and Nyquist | Higher Q → narrower passband |
| `notch` | `cutoff`, `Q` | DC and Nyquist | Narrow band at cutoff | Higher Q → narrower notch |
| `allpass` | `cutoff`, `Q` | All (unity magnitude) | Nothing | Higher Q → steeper phase transition |

---

## Biquad structure

A biquad (second-order section) transfer function:

```text
        b0 + b1·z⁻¹ + b2·z⁻²
H(z) = ─────────────────────────
         1 + a1·z⁻¹ + a2·z⁻²
```

Shared preliminary computation (all five filters share the same a coefficients):

```text
ω₀     = 2π · cutoff / fs
α      = sin(ω₀) / (2Q)
cos_w0 = cos(ω₀)
a0     = 1 + α
```

---

## `lowpass`

**Coefficients:**

```text
b = [(1 − cos_w0)/2,  (1 − cos_w0),  (1 − cos_w0)/2]  / a0
a = [1.0,             −2·cos_w0,      (1 − α)]          / a0
```

**Response:** H(0) = 1 (DC pass), H(π) = 0 (Nyquist null). −3 dB at `cutoff` when Q = 0.707 (Butterworth). Roll-off −40 dB/decade.

---

## `highpass`

**Coefficients:**

```text
b = [(1 + cos_w0)/2,  −(1 + cos_w0),  (1 + cos_w0)/2]  / a0
a = [1.0,              −2·cos_w0,       (1 − α)]         / a0
```

**Response:** H(0) = 0 (DC null), H(π) = 1 (Nyquist pass). Same poles as lowpass; only zero placement differs.

---

## `bandpass`

Constant 0 dB peak gain variant (Audio EQ Cookbook).

**Coefficients:**

```text
b = [α,    0,  −α]  / a0
a = [1.0,  −2·cos_w0,  (1 − α)]  / a0
```

**Response:** H(0) = 0, H(π) = 0, H(e^jω₀) = 1. Bandwidth = `cutoff / Q`. Higher Q → narrower passband.

---

## `notch`

**Coefficients:**

```text
b = [1,          −2·cos_w0,  1        ]  / a0
a = [1.0,        −2·cos_w0,  (1 − α)  ]  / a0
```

**Response:** H(0) = 1, H(π) = 1, H(e^jω₀) = 0. No roll-off; returns to unity gain above and below the notch. Higher Q → narrower notch.

---

## `allpass`

**Coefficients:**

```text
b = [(1 − α),  −2·cos_w0,  (1 + α)]  / a0
a = [1.0,       −2·cos_w0/a0,  (1−α)/a0]
```

The b coefficients are the time-reverse of a, guaranteeing |H(z)| = 1 on the unit circle.

**Response:** |H(e^jω)| = 1 for all ω. Phase rotates from 0° at DC to −360° at Nyquist; steepest transition at `cutoff`. Higher Q → taller, narrower group-delay peak.

---

## Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `cutoff` | 1–21000 Hz (at 44100 Hz `fs`) | Must satisfy 0 < cutoff < fs/2; values outside this range produce undefined biquad behaviour |
| `Q` | 0.1–20 | Controls resonance width; Q = 0.707 gives Butterworth maximally-flat response; Q < 0.5 overdamps (no resonance peak); Q > 5 produces a narrow, tall resonance peak |
| `fs` | 8000–192000 Hz | Coefficients are re-derived per call; any standard audio sample rate is valid |
