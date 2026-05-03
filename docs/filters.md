# Biquad Filters

All filters live in `src/python/filters.py`. Each takes a signal array and returns a filtered array of the same length. They are implemented as second-order IIR (Infinite Impulse Response) filters using the **biquad** topology, following the [Audio EQ Cookbook](https://www.musicdsp.org/en/latest/Filters/197-rbj-audio-eq-cookbook.html) coefficient formulas.

Filters are organized into three types:

| Type | Filters |
| --- | --- |
| **EQ / parametric** | `lowpass`, `highpass`, `bandpass`, `notch`, `allpass` |
| **Shelving** | `lowshelf`, `highshelf` |
| **Utility** | `dc_block` |

---

## The biquad structure

A biquad (second-order section) is the fundamental building block of digital IIR filters. Its transfer function in the Z-domain is:

```text
        b0 + b1·z⁻¹ + b2·z⁻²
H(z) = ─────────────────────────
         1 + a1·z⁻¹ + a2·z⁻²
```

The difference equation (what actually runs sample-by-sample) is:

```text
y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2]
     −  a1·y[n−1] − a2·y[n−2]
```

The **b** coefficients are the feedforward (FIR) part; the **a** coefficients are the feedback (IIR) part that creates the resonance and sharp roll-off not achievable with a finite-length filter.

A second-order biquad has two poles and up to two zeros in the Z-plane. The frequency response is completely determined by where the poles and zeros sit on or inside the unit circle. A pole near the unit circle at angle `ω₀` creates a peak in the response near that frequency; a zero on the unit circle at angle `ω₀` creates a null (exactly zero gain) at that frequency.

`scipy.signal.lfilter` is used to apply the coefficients; it implements the Direct Form II transposed structure internally, which is numerically stable for second-order sections.

---

## Shared coefficient derivation

The five Q-based filters (low-pass, high-pass, band-pass, notch, all-pass) share the same preliminary computation (Audio EQ Cookbook nomenclature). Shelf filters use a different α formula; see their sections below.

```text
ω₀     = 2π · cutoff / fs          # normalized angular cutoff frequency
α      = sin(ω₀) / (2Q)            # bandwidth parameter
cos_w0 = cos(ω₀)
a0     = 1 + α                     # normalization denominator
```

The `a0` denominator is divided out before passing coefficients to `lfilter`, yielding a monic `a[0] = 1.0`.

The Q factor controls the width of the transition around `ω₀`. At `Q = 1/(2α)`, the poles sit on a circle of radius `r = sqrt(1 − α²/(1+α²))` ≈ `1 − α` for small α. Higher Q moves poles closer to the unit circle, sharpening the transition and (for bandpass) narrowing the passband.

---

## EQ / parametric

Five filters that shape the frequency response by passing, attenuating, or phase-shifting specific bands. All share the same feedback (a) coefficients — only the feedforward (b) coefficients differ, controlling zero placement.

| Filter | Key parameters | Passes | Rejects | Q effect |
| --- | --- | --- | --- | --- |
| `lowpass` | `cutoff`, `Q` | Below cutoff | Above cutoff | Higher Q → resonant peak below cutoff |
| `highpass` | `cutoff`, `Q` | Above cutoff | Below cutoff | Higher Q → resonant peak above cutoff |
| `bandpass` | `cutoff`, `Q` | Band around cutoff | DC and Nyquist | Higher Q → narrower passband |
| `notch` | `cutoff`, `Q` | DC and Nyquist | Narrow band at cutoff | Higher Q → narrower notch |
| `allpass` | `cutoff`, `Q` | All (unity magnitude) | Nothing | Higher Q → steeper phase transition |

### Low-pass filter (`lowpass`)

**Purpose:** Passes frequencies below `cutoff`; attenuates frequencies above it.

#### Low-pass coefficient set

```text
b = [(1 − cos_w0)/2,  (1 − cos_w0),  (1 − cos_w0)/2]  / a0
a = [1.0,             −2·cos_w0,      (1 − α)]          / a0
```

#### Low-pass frequency response

- **H(0) = 1** (DC, z = 1): both numerator and denominator evaluate to `(1 − cos_w0)` terms that cancel to 1, confirming unity gain at DC.
- **H(π) ≈ 0** (Nyquist, z = −1): the numerator terms combine to zero when all `z⁻¹ = −1`.
- **−3 dB point** at `ω₀` when `Q = 1/√2 ≈ 0.707` (Butterworth, maximally flat).
- Higher Q creates a resonant peak just below `cutoff` before rolling off.
- Roll-off is **−40 dB/decade** (−12 dB/octave) asymptotically in the stopband, which is the defining property of a 2nd-order filter.

The zeros are at z = −1 (Nyquist), creating a null there.

---

### High-pass filter (`highpass`)

**Purpose:** Attenuates frequencies below `cutoff`; passes frequencies above it. Rejects DC.

#### High-pass coefficient set

```text
b = [(1 + cos_w0)/2,  −(1 + cos_w0),  (1 + cos_w0)/2]  / a0
a = [1.0,              −2·cos_w0,       (1 − α)]         / a0
```

The feedback (a) coefficients are identical to the low-pass filter — same poles, same resonance shape. Only the feedforward (b) coefficients differ; the sign flip on b1 moves the zeros from Nyquist to DC.

#### High-pass frequency response

- **H(0) = 0** (DC, z = 1): b0 + b1 + b2 = `(1+c)/2 − (1+c) + (1+c)/2 = 0`. Zeros are at z = +1 (DC).
- **H(π) = 1** (Nyquist): numerator and denominator evaluate to `(1 + cos_w0)` terms that cancel to 1.
- **−3 dB** at `ω₀` for `Q = 0.707`, same Butterworth condition as low-pass.
- **−40 dB/decade** roll-off into the low-frequency stopband.

The LP and HP coefficient sets are duals of each other under the substitution `cos_w0 → −cos_w0` in the b numerator, reflecting the spectral flip `ω → π − ω`.

---

### Band-pass filter (`bandpass`)

**Purpose:** Passes a band of frequencies centered on `cutoff`; attenuates both below and above. Rejects DC. Bandwidth = `cutoff / Q`.

#### Band-pass coefficient set (constant 0 dB peak gain)

```text
b = [α,    0,  −α]  / a0
a = [1.0,  −2·cos_w0,  (1 − α)]  / a0
```

where `α = sin(ω₀) / (2Q)`.

#### Two bandpass variants

The Audio EQ Cookbook defines two bandpass variants:

| Variant | b0 | Peak gain |
| --- | --- | --- |
| Constant 0 dB peak gain | `α` | 1 (0 dB) regardless of Q |
| Constant skirt gain | `sin(ω₀)/2 = Q·α` | Q (scales with Q) |

This implementation uses the **constant 0 dB peak gain** variant. Using the skirt-gain variant is a common mistake: the peak gain of Q means that at high Q the signal at the center frequency is amplified by a factor of Q, making the output incompatible with downstream stages that expect unity-gain filtering.

#### Band-pass frequency response

- **H(0) = 0**: b0 + b1 + b2 = `α + 0 − α = 0`. Zero at z = +1 (DC).
- **H(π) = 0**: at z = −1, numerator = `α·(−1)⁰ + 0 − α·(−1)² = α − α = 0`. Zero at z = −1 (Nyquist).
- **H(e^jω₀) = 1**: at the center frequency the response achieves exactly 0 dB (unity gain).
- **Q = f₀ / BW**: higher Q narrows the passband. Q = 0.5 gives a very wide (nearly all-pass) band; Q = 10 gives a narrow resonant peak.
- The poles are the same as LP and HP; only the zeros differ (both placed on the unit circle at DC and Nyquist to ensure zero gain there).

#### Bandwidth interpretation

At −3 dB, the two edges of the passband are approximately at:

```text
f_lower ≈ f₀ · (√(1 + 1/(4Q²)) − 1/(2Q))
f_upper ≈ f₀ · (√(1 + 1/(4Q²)) + 1/(2Q))
BW = f_upper − f_lower ≈ f₀ / Q
```

For high Q the approximation `BW ≈ f₀ / Q` is tight; for low Q (< 1) the asymmetry of the log-frequency scale means the geometric center `√(f_lower · f_upper) = f₀` is preserved but the arithmetic bandwidth is only approximately `f₀ / Q`.

---

### Notch filter (`notch`)

**Purpose:** Rejects a narrow band of frequencies centered on `cutoff`; passes all other frequencies at unity gain, including DC and Nyquist.

#### Notch coefficient set

```text
b = [1,          −2·cos_w0,  1        ]  / a0
a = [1.0,        −2·cos_w0,  (1 − α)  ]  / a0
```

The denominator (a) is identical to LP, HP, and BP — same poles, same resonance structure. Only the numerator differs: the notch numerator lacks the scaling applied to b0 and b2 in the other filters, placing zeros at both DC and Nyquist simultaneously removed so that only the target frequency is rejected.

#### Notch frequency response

- **H(0) = 1** (DC, z = 1): b0 + b1 + b2 = `1 − 2cos_w0 + 1` and a0 + a1 + a2 = `(1+α) − 2cos_w0 + (1−α)` = `2 − 2cos_w0`; both sides are equal, giving unity gain at DC.
- **H(π) = 1** (Nyquist, z = −1): b0 − b1 + b2 = `1 + 2cos_w0 + 1` and denominator = `2 + 2cos_w0`; equal, giving unity gain at Nyquist.
- **H(e^jω₀) = 0**: zero exactly on the unit circle at the cutoff frequency.
- **Q controls notch width**: high Q gives a narrow notch (adjacent frequencies pass with little loss); low Q gives a wide notch.
- No roll-off; the filter returns to unity gain above and below the notch.

---

### All-pass filter (`allpass`)

**Purpose:** Passes all frequencies at exactly unity magnitude. Only the phase changes — the filter is invisible to magnitude-based analysis but affects time alignment and phase-sensitive signal paths.

#### All-pass coefficient set

```text
b = [(1 − α),  −2·cos_w0,  (1 + α)]  / a0   =   [(1−α)/a0,  −2·cos_w0/a0,  1.0]
a = [1.0,       −2·cos_w0/a0,        (1−α)/a0]
```

A key property: `b` is the **time-reverse of `a`** — `b[k] = a[2−k]` after normalization. This guarantees `|H(z)| = 1` on the unit circle: the numerator and denominator polynomials are mirror images, so their magnitudes cancel everywhere. Note that after normalization `b[2] = (1+α)/a0 = 1.0` exactly.

#### All-pass frequency response

- **|H(e^jω)| = 1** for all ω: no frequency is attenuated or boosted.
- **Phase** rotates from 0° at DC to −360° at Nyquist, with the steepest transition at `ω₀`.
- **Group delay** peaks at `ω₀` with value `2Q / (π·f₀)` seconds. Signals near `ω₀` are delayed more than signals far from it.
- **Q controls transition sharpness**: higher Q gives a steeper phase transition and a narrower, taller group-delay peak.

---

## Shelving

Two filters that boost or cut a broad frequency region (low or high end) by a fixed amount in dB, tapering smoothly through a transition band around `cutoff`.

| Filter | Key parameters | Boosts/cuts | Unity gain at |
| --- | --- | --- | --- |
| `lowshelf` | `cutoff`, `gain_db` | Below cutoff | Nyquist |
| `highshelf` | `cutoff`, `gain_db` | Above cutoff | DC |

Shelf filters use `gain_db` instead of Q. Positive `gain_db` adds gain at the shelved end; negative cuts it. Both use shelf slope S = 1 (steepest monotonic slope), with the α formula:

```text
A      = 10^(gain_db / 40)     # linear amplitude at the shelved end; A² = 10^(gain_db/20)
α      = sin(ω₀) / √2          # S = 1 simplification
√A     = sqrt(A)
```

### Low-shelf filter (`lowshelf`)

**Purpose:** Boosts or cuts frequencies below `cutoff` by `gain_db` dB, while leaving frequencies above `cutoff` at unity gain.

#### Low-shelf coefficient set

```text
b0 =    A · [(A+1) − (A−1)·cos_w0 + 2·√A·α]
b1 =  2·A · [(A−1) − (A+1)·cos_w0          ]
b2 =    A · [(A+1) − (A−1)·cos_w0 − 2·√A·α]
a0 =        (A+1) + (A−1)·cos_w0 + 2·√A·α
a1 =   −2 · [(A−1) + (A+1)·cos_w0          ]
a2 =         (A+1) + (A−1)·cos_w0 − 2·√A·α
```

All coefficients are divided by a0 before use.

#### Low-shelf frequency response

- **H(0) = A²** (DC, z = 1): numerator sums to `4A²(1 − cos_w0)` and denominator to `4(1 − cos_w0)`, giving gain A². In dB: `20·log10(A²) = gain_db`.
- **H(π) = 1** (Nyquist, z = −1): numerator sums to `4A(1 + cos_w0)` and denominator to `4A(1 + cos_w0)`, giving unity gain. The shelf only affects the low-frequency end.
- Transition is centered on `cutoff`; the half-gain point (in dB) is at `cutoff`.
- `gain_db = 0` → identity (coefficients reduce to all-pass with unity gain everywhere).

---

### High-shelf filter (`highshelf`)

**Purpose:** Boosts or cuts frequencies above `cutoff` by `gain_db` dB, while leaving frequencies below `cutoff` at unity gain.

The high-shelf is the spectral dual of the low-shelf, obtained by negating `cos_w0` in the denominator and reversing sign conventions in the numerator.

#### High-shelf coefficient set

```text
b0 =    A · [(A+1) + (A−1)·cos_w0 + 2·√A·α]
b1 = −2·A · [(A−1) + (A+1)·cos_w0          ]
b2 =    A · [(A+1) + (A−1)·cos_w0 − 2·√A·α]
a0 =        (A+1) − (A−1)·cos_w0 + 2·√A·α
a1 =    2 · [(A−1) − (A+1)·cos_w0          ]
a2 =         (A+1) − (A−1)·cos_w0 − 2·√A·α
```

A and α are defined identically to the low-shelf. All coefficients are divided by a0.

#### High-shelf frequency response

- **H(0) = 1** (DC, z = 1): numerator sums to `4A(1 − cos_w0)` and denominator to `4A(1 − cos_w0)`, giving unity gain at DC.
- **H(π) = A²** (Nyquist, z = −1): numerator sums to `4A²(1 + cos_w0)` and denominator to `4(1 + cos_w0)`, giving gain A². In dB: `gain_db`.
- `gain_db > 0` → boost above `cutoff`; `gain_db < 0` → cut above `cutoff`.

---

## Utility

### DC blocking filter (`dc_block`)

**Purpose:** Removes DC offset and sub-sonic content below `cutoff` Hz while leaving all higher-frequency content unchanged. Unlike the biquad high-pass filter, `dc_block` is first-order (one pole, one zero) and is specifically designed for DC removal rather than audio-band shaping.

#### Coefficient derivation

`dc_block` applies the bilinear transform to the first-order analog high-pass H_a(s) = s / (s + ω_c):

```text
k      = tan(π · cutoff / fs)     # bilinear pre-warp
b      = [1/(1+k),  −1/(1+k)]
a      = [1.0,      −(1−k)/(1+k)]
```

The bilinear transform maps the analog −3 dB frequency ω_c exactly to the digital frequency `cutoff` Hz, with no frequency-axis warping error at the cutoff.

#### Frequency response

- **H(1) = 0** (DC, z = 1): b0 + b1 = `1/(1+k) − 1/(1+k) = 0`. The zero at z = +1 is a structural property independent of `cutoff`; DC is always completely rejected.
- **H(−1) = 1** (Nyquist, z = −1): numerator = `2/(1+k)`, denominator = `1 + (1−k)/(1+k) = 2/(1+k)`; these cancel to unity for any `cutoff`.
- **−3 dB at `cutoff`**: the bilinear pre-warp guarantees the half-power point is exactly at the requested frequency.
- **−20 dB/decade** rolloff below `cutoff` (first-order slope).

#### Comparison with `highpass`

| Property | `highpass` | `dc_block` |
| --- | --- | --- |
| Order | 2nd (biquad) | 1st |
| Roll-off | −40 dB/decade | −20 dB/decade |
| Q control | Yes | No |
| DC rejection | Complete | Complete (structural zero) |
| Typical cutoff | 20 Hz – audio band | < 20 Hz (sub-sonic) |

Use `dc_block` when you only need DC and hum removal and want the flattest possible response in the audio band. Use `highpass` when you need steeper roll-off or Q control for audio filtering.
