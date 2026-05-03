# Nonlinear

`src/python/nonlinear.py` implements four memoryless nonlinear waveshaping functions for distortion, saturation, and quantisation.

## Summary

| Function | Key parameters | Output range | Harmonics generated |
| --- | --- | --- | --- |
| `hard_clip` | `threshold` | [−threshold, +threshold] | All harmonics |
| `soft_clip` | `drive` | (−1, +1) | All harmonics (odd-dominant) |
| `waveshape` | `coeffs` | unbounded | Determined by polynomial order |
| `bitcrush` | `bits` | quantised levels | All harmonics |

---

### `hard_clip`

Symmetrically clips the signal to the interval [−threshold, +threshold]:

```text
y[n] = clip(x[n], −T, +T)
```

Clipping introduces discontinuities in the waveform, producing a rich spectrum of all harmonics (both even and odd). With a high threshold the signal passes unchanged; as the threshold decreases, more cycles are squared-off. Commonly used in guitar amp emulations and to prevent digital overload.

**Spectrum:** the clipped waveform can be decomposed into Fourier series. For a symmetrically clipped sine wave the harmonics follow a pattern of odd and even partials depending on the clip depth.

---

### `soft_clip`

Hyperbolic tangent saturation:

```text
y[n] = tanh(drive · x[n])
```

The output is bounded in (−1, +1) for all inputs. Near zero the transfer function is approximately linear (tanh(x) ≈ x). As the input grows the gain compresses, producing a smooth transition from clean to saturated. Higher `drive` values move the knee lower, increasing harmonic distortion at lower input levels.

**Odd symmetry:** tanh(−x) = −tanh(x), so only odd harmonics are generated for a pure sine input (3rd, 5th, 7th, …). This gives a "tube-like" character compared to the even harmonics of asymmetric distortions.

---

### `waveshape`

General-purpose polynomial waveshaping:

```text
y[n] = c₀ + c₁·x[n] + c₂·x[n]² + c₃·x[n]³ + …
```

`coeffs = [c₀, c₁, c₂, …]` in ascending power order. The generated harmonics correspond directly to the polynomial terms: a coefficient at degree k produces the k-th harmonic. Setting only odd-degree coefficients (c₁, c₃, c₅, …) gives odd-only distortion; including c₂, c₄, … adds even harmonics.

**Use case:** Chebyshev polynomials give exact single-harmonic generation. For example, the 3rd Chebyshev polynomial T₃(x) = 4x³ − 3x generates a pure 3rd harmonic from a full-scale sine.

---

### `bitcrush`

Reduces the amplitude resolution to `bits` bits by rounding to `2^bits` quantisation levels:

```text
y[n] = round(x[n] · 2^(bits−1)) / 2^(bits−1)
```

At `bits = 16` the distortion is inaudible; at `bits = 8` (256 levels) quantisation noise is audible; at `bits = 4` (16 levels) strong harmonic distortion is clearly present. At `bits = 1` the output is a binary signal (−1 or +1).

**Spectrum:** the quantisation error acts as a periodic deterministic signal for periodic inputs, adding harmonics at multiples of the input frequency. For noise-like inputs, the quantisation error approaches additive white noise.
