# Delay Line

## `delay_line`

A causal integer-sample delay:

```text
y[n] = x[n − D]
```

Transfer function: H(z) = z^(−D). All samples before index D are zero (initial rest). This is the simplest possible DSP operation and serves as the building block for the two comb filters (`feedback_delay`, `comb_filter`). It is also used by `spatial.haas`.

**Parameters:** `delay_samples` — integer delay D in samples.

**Use cases:** time-aligning signals, creating echo taps, building more complex delay networks.

---

## `fractional_delay_line`

Extends `delay_line` to non-integer delays using Lagrange polynomial interpolation. Given a delay d = D + f where D = floor(d) and f ∈ [0, 1):

```text
y[n] = Σ_k  w_k · x[n − D + k]
```

The weights w_k are evaluated from the Lagrange basis polynomials at position −f relative to sample n − D. Because the weights always sum to 1, the filter preserves DC (gain = 1 at ω = 0). In the frequency domain the filter approximates a linear-phase all-pass — magnitude ≈ 1 across most of the audible band, with a phase shift proportional to the fractional delay f.

**Parameters:**

| Parameter | Type | Notes |
| --- | --- | --- |
| `signal` | array | Input signal |
| `delay_samples` | float or array | Non-negative delay in samples. A 1-D array applies a different delay to each output sample (used for vibrato, chorus, flanger). |
| `order` | int (3 or 4) | Lagrange polynomial degree. Order 3 uses a 4-point stencil; order 4 uses a 5-point symmetric stencil. Default: 3. |

### Lagrange weights (order 3)

The stencil reads from positions n−D+1, n−D, n−D−1, n−D−2 with weights derived from the degree-3 Lagrange basis evaluated at x = −f:

```text
w₀ = −f(1−f)(2−f) / 6          [x[n−D+1]]
w₁ = (1+f)(1−f)(2−f) / 2       [x[n−D]  ]
w₂ =  f(1+f)(2−f) / 2          [x[n−D−1]]
w₃ = −f(1+f)(1−f) / 6          [x[n−D−2]]
```

At f = 0: w₁ = 1, all others zero — reduces to `delay_line(signal, D)`. At f = 0.5: w₀ = w₃ = −1/16, w₁ = w₂ = 9/16 — symmetric half-sample interpolation.

### Lagrange weights (order 4)

The stencil reads symmetrically from positions n−D+2 … n−D−2 with weights from the degree-4 basis at x = −f:

```text
w₀ =  (2−f)(1−f)f(1+f) / 24    [x[n−D+2]]
w₁ = −(2−f)(1−f)f(2+f) / 6     [x[n−D+1]]
w₂ =  (2−f)(1−f)(1+f)(2+f) / 4 [x[n−D]  ]
w₃ =  (2−f)f(1+f)(2+f) / 6     [x[n−D−1]]
w₄ = −(1−f)f(1+f)(2+f) / 24    [x[n−D−2]]
```

The 5-point stencil has lower approximation error for high-frequency content and is preferred when a chorus or flanger sweeps into high modulation depths. The trade-off is a slightly wider stencil that increases the effective group delay by one sample on the leading edge.

### Modulated (per-sample) delays

When `delay_samples` is a 1-D array of length N, each output sample uses its own integer+fractional split. The padding is computed from the maximum integer delay across the entire array, so the operation remains a single vectorised NumPy pass with no Python loop.

**Use cases:** vibrato and pitch modulation (sinusoidal modulation, 1–30 ms depth), chorus (10–30 ms, shallow modulation), flanger (0.5–5 ms, deep modulation with feedback), Karplus-Strong (fixed fractional delay to hit exact pitch).
