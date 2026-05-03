# Biquad Filters

All filters live in `src/python/filters.py`. Each takes a signal array and returns a filtered array of the same length. They are implemented as second-order IIR (Infinite Impulse Response) filters using the **biquad** topology, following the [Audio EQ Cookbook](https://www.musicdsp.org/en/latest/Filters/197-rbj-audio-eq-cookbook.html) coefficient formulas.

---

## The biquad structure

A biquad (second-order section) is the fundamental building block of digital IIR filters. Its transfer function in the Z-domain is:

```
        b0 + b1·z⁻¹ + b2·z⁻²
H(z) = ─────────────────────────
         1 + a1·z⁻¹ + a2·z⁻²
```

The difference equation (what actually runs sample-by-sample) is:

```
y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2]
     −  a1·y[n−1] − a2·y[n−2]
```

The **b** coefficients are the feedforward (FIR) part; the **a** coefficients are the feedback (IIR) part that creates the resonance and sharp roll-off not achievable with a finite-length filter.

A second-order biquad has two poles and up to two zeros in the Z-plane. The frequency response is completely determined by where the poles and zeros sit on or inside the unit circle. A pole near the unit circle at angle `ω₀` creates a peak in the response near that frequency; a zero on the unit circle at angle `ω₀` creates a null (exactly zero gain) at that frequency.

`scipy.signal.lfilter` is used to apply the coefficients; it implements the Direct Form II transposed structure internally, which is numerically stable for second-order sections.

---

## Shared coefficient derivation

All three filters share the same preliminary computation (Audio EQ Cookbook nomenclature):

```
ω₀    = 2π · cutoff / fs          # normalized angular cutoff frequency
α     = sin(ω₀) / (2Q)            # bandwidth parameter
cos_w0 = cos(ω₀)
a0    = 1 + α                     # normalization denominator
```

The `a0` denominator is divided out before passing coefficients to `lfilter`, yielding a monic `a[0] = 1.0`.

The Q factor controls the width of the transition around `ω₀`. At `Q = 1/(2α)`, the poles sit on a circle of radius `r = sqrt(1 − α²/(1+α²))` ≈ `1 − α` for small α. Higher Q moves poles closer to the unit circle, sharpening the transition and (for bandpass) narrowing the passband.

---

## Low-pass filter (`lowpass`)

**Purpose:** Passes frequencies below `cutoff`; attenuates frequencies above it.

### Coefficient set

```
b = [(1 − cos_w0)/2,  (1 − cos_w0),  (1 − cos_w0)/2]  / a0
a = [1.0,             −2·cos_w0,      (1 − α)]          / a0
```

### Frequency response behavior

- **H(0) = 1** (DC, z = 1): both numerator and denominator evaluate to `(1 − cos_w0)` terms that cancel to 1, confirming unity gain at DC.
- **H(π) ≈ 0** (Nyquist, z = −1): the numerator terms combine to zero when all `z⁻¹ = −1`.
- **−3 dB point** at `ω₀` when `Q = 1/√2 ≈ 0.707` (Butterworth, maximally flat).
- Higher Q creates a resonant peak just below `cutoff` before rolling off.
- Roll-off is **−40 dB/decade** (−12 dB/octave) asymptotically in the stopband, which is the defining property of a 2nd-order filter.

The zeros are at z = −1 (Nyquist), creating a null there.

---

## High-pass filter (`highpass`)

**Purpose:** Attenuates frequencies below `cutoff`; passes frequencies above it. Rejects DC.

### Coefficient set

```
b = [(1 + cos_w0)/2,  −(1 + cos_w0),  (1 + cos_w0)/2]  / a0
a = [1.0,              −2·cos_w0,       (1 − α)]         / a0
```

The feedback (a) coefficients are identical to the low-pass filter — same poles, same resonance shape. Only the feedforward (b) coefficients differ; the sign flip on b1 moves the zeros from Nyquist to DC.

### Frequency response behavior

- **H(0) = 0** (DC, z = 1): b0 + b1 + b2 = `(1+c)/2 − (1+c) + (1+c)/2 = 0`. Zeros are at z = +1 (DC).
- **H(π) = 1** (Nyquist): numerator and denominator evaluate to `(1 + cos_w0)` terms that cancel to 1.
- **−3 dB** at `ω₀` for `Q = 0.707`, same Butterworth condition as low-pass.
- **−40 dB/decade** roll-off into the low-frequency stopband.

The LP and HP coefficient sets are duals of each other under the substitution `cos_w0 → −cos_w0` in the b numerator, reflecting the spectral flip `ω → π − ω`.

---

## Band-pass filter (`bandpass`)

**Purpose:** Passes a band of frequencies centered on `cutoff`; attenuates both below and above. Rejects DC. Bandwidth = `cutoff / Q`.

### Coefficient set (constant 0 dB peak gain)

```
b = [α,    0,  −α]  / a0
a = [1.0,  −2·cos_w0,  (1 − α)]  / a0
```

where `α = sin(ω₀) / (2Q)`.

### Two bandpass variants

The Audio EQ Cookbook defines two bandpass variants:

| Variant | b0 | Peak gain |
|---|---|---|
| Constant 0 dB peak gain | `α` | 1 (0 dB) regardless of Q |
| Constant skirt gain | `sin(ω₀)/2 = Q·α` | Q (scales with Q) |

This implementation uses the **constant 0 dB peak gain** variant. Using the skirt-gain variant is a common mistake: the peak gain of Q means that at high Q the signal at the center frequency is amplified by a factor of Q, making the output incompatible with downstream stages that expect unity-gain filtering.

### Frequency response behavior

- **H(0) = 0**: b0 + b1 + b2 = `α + 0 − α = 0`. Zero at z = +1 (DC).
- **H(π) = 0**: at z = −1, numerator = `α·(−1)⁰ + 0 − α·(−1)² = α − α = 0`. Zero at z = −1 (Nyquist).
- **H(e^jω₀) = 1**: at the center frequency the response achieves exactly 0 dB (unity gain).
- **Q = f₀ / BW**: higher Q narrows the passband. Q = 0.5 gives a very wide (nearly all-pass) band; Q = 10 gives a narrow resonant peak.
- The poles are the same as LP and HP; only the zeros differ (both placed on the unit circle at DC and Nyquist to ensure zero gain there).

### Bandwidth interpretation

At −3 dB, the two edges of the passband are approximately at:

```
f_lower ≈ f₀ · (√(1 + 1/(4Q²)) − 1/(2Q))
f_upper ≈ f₀ · (√(1 + 1/(4Q²)) + 1/(2Q))
BW = f_upper − f_lower ≈ f₀ / Q
```

For high Q the approximation `BW ≈ f₀ / Q` is tight; for low Q (< 1) the asymmetry of the log-frequency scale means the geometric center `√(f_lower · f_upper) = f₀` is preserved but the arithmetic bandwidth is only approximately `f₀ / Q`.
