# Waveshaping

## `waveshape`

General-purpose polynomial waveshaping:

```text
y[n] = c₀ + c₁·x[n] + c₂·x[n]² + c₃·x[n]³ + …
```

`coeffs = [c₀, c₁, c₂, …]` in ascending power order. The generated harmonics correspond directly to the polynomial terms: a coefficient at degree k produces the k-th harmonic. Setting only odd-degree coefficients (c₁, c₃, c₅, …) gives odd-only distortion; including c₂, c₄, … adds even harmonics.

**Use case:** Chebyshev polynomials give exact single-harmonic generation. For example, the 3rd Chebyshev polynomial T₃(x) = 4x³ − 3x generates a pure 3rd harmonic from a full-scale sine.

---

## `bitcrush`

Reduces the amplitude resolution to `bits` bits by rounding to `2^bits` quantisation levels:

```text
y[n] = round(x[n] · 2^(bits−1)) / 2^(bits−1)
```

At `bits = 16` the distortion is inaudible; at `bits = 8` (256 levels) quantisation noise is audible; at `bits = 4` (16 levels) strong harmonic distortion is clearly present. At `bits = 1` the output is a binary signal (−1 or +1).

**Spectrum:** the quantisation error acts as a periodic deterministic signal for periodic inputs, adding harmonics at multiples of the input frequency. For noise-like inputs, the quantisation error approaches additive white noise.
