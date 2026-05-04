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
