# Shelving Filters

Two biquad filters that boost or cut a broad frequency region by a fixed amount in dB, tapering smoothly through a transition band around `cutoff`.

| Filter | Key parameters | Boosts/cuts | Unity gain at |
| --- | --- | --- | --- |
| `lowshelf` | `cutoff`, `gain_db` | Below cutoff | Nyquist |
| `highshelf` | `cutoff`, `gain_db` | Above cutoff | DC |

Positive `gain_db` adds gain at the shelved end; negative cuts it. Both use shelf slope S = 1:

```text
A  = 10^(gain_db / 40)    # A² = linear gain at shelved end
α  = sin(ω₀) / √2         # S = 1 simplification
```

---

## `lowshelf`

**Coefficients:**

```text
b0 =    A · [(A+1) − (A−1)·cos_w0 + 2·√A·α]
b1 =  2·A · [(A−1) − (A+1)·cos_w0          ]
b2 =    A · [(A+1) − (A−1)·cos_w0 − 2·√A·α]
a0 =        (A+1) + (A−1)·cos_w0 + 2·√A·α
a1 =   −2 · [(A−1) + (A+1)·cos_w0          ]
a2 =         (A+1) + (A−1)·cos_w0 − 2·√A·α
```

**Response:**

- H(0) = A² → DC gain equals `gain_db` dB
- H(π) = 1 → Nyquist always passes unchanged
- Transition centered on `cutoff`; `gain_db = 0` → identity

---

## `highshelf`

The spectral dual of lowshelf (negate `cos_w0` in denominator, reverse sign conventions in numerator).

**Coefficients:**

```text
b0 =    A · [(A+1) + (A−1)·cos_w0 + 2·√A·α]
b1 = −2·A · [(A−1) + (A+1)·cos_w0          ]
b2 =    A · [(A+1) + (A−1)·cos_w0 − 2·√A·α]
a0 =        (A+1) − (A−1)·cos_w0 + 2·√A·α
a1 =    2 · [(A−1) − (A+1)·cos_w0          ]
a2 =         (A+1) − (A−1)·cos_w0 − 2·√A·α
```

**Response:**

- H(0) = 1 → DC always passes unchanged
- H(π) = A² → Nyquist gain equals `gain_db` dB
- `gain_db > 0` → boost above `cutoff`; `gain_db < 0` → cut above `cutoff`

---

## Parameter Ranges

| Parameter | Range | Notes |
| --- | --- | --- |
| `cutoff` | 20–10000 Hz (at 44100 Hz `fs`) | Transition band centre; must satisfy 0 < cutoff < fs/2 |
| `gain_db` | −18 to +18 dB | Values outside ±18 dB are mathematically valid but produce large linear gains (A² = 10^(gain_db/20)); `gain_db = 0` is the identity (passes signal unchanged) |
| `fs` | 8000–192000 Hz | Any standard audio sample rate |
